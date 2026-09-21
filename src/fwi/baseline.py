"""Pure velocity-gradient multiscale FWI with the same acquisition and initial model."""
from __future__ import annotations
import copy
import time
from pathlib import Path
import numpy as np
import torch
from src.zbank.initializer import training_latents
from src.models.autoencoder import load_autoencoder
from src.environment.scheduler import EnvironmentScheduler
from src.physics.misfit import MultiFrequencyMisfit
from src.physics.wave_solver import SolverCounters
from src.inversion.problem import make_problem,make_solver,Problem
from src.inversion.evaluator import smooth_velocity
from src.inversion.stopping import Stopping
from src.fwi.optimizer import make_optimizer
from src.utils.config import load_config,merge,storage_path,save_config
from src.utils.seed import setup,rng_state,restore_rng
from src.utils.logger import write_json,write_jsonl
from src.utils.metrics import velocity_metrics
from src.utils.visualization import plot_results


def run_baseline(config: dict,output: str | Path | None = None,plots: bool = True,
                 resume: str | Path | None = None,max_additional_generations: int | None = None) -> dict:
    """Optimize every physical velocity cell, without GA, bank or latent QC.

    Uses the same AE-derived initial velocity as hybrid runs, then leaves the
    learned manifold. Fixed-reference waveform loss selects its final result.
    """
    device = setup(config["seed"],config.get("device","auto"),config.get("threads",2))
    directory = storage_path(output or str(config["output"])+"_gradient")
    directory.mkdir(parents=True,exist_ok=True)
    env_config = merge(load_config(config["environment_config"]),config.get("environment_overrides",{}))
    scheduler = EnvironmentScheduler(env_config,config["generations"],True)
    final_environment = scheduler.at_progress(1.)
    state = torch.load(resume,map_location="cpu",weights_only=True) if resume else None
    if state:
        old,new = copy.deepcopy(state["config"]),copy.deepcopy(config)
        old.pop("output",None); new.pop("output",None)
        if old != new or env_config != state["environment_config"]:
            raise ValueError("Baseline resume config differs")
        problem = Problem.from_state_dict(state["problem"],device)
        vmin,vmax = state["bounds"]
    else:
        model,training_config = load_autoencoder(storage_path(config["checkpoint"]),device)
        model.freeze()
        prior = training_latents(model,training_config)
        with torch.no_grad():
            initial = model.decode(prior.mean(0)[None],tuple(config["shape"]))
        problem = make_problem(config,initial,device)
        vmin,vmax = model.velocity_min,model.velocity_max
    q = ((problem.initial_velocity-vmin)/(vmax-vmin)).detach().clone().requires_grad_()
    if state:
        with torch.no_grad():
            q.copy_(state["q"].to(device))
    baseline_config = config.get("baseline",{})
    optimizer_name = baseline_config.get("optimizer","Adam")
    optimizer = make_optimizer([q],optimizer_name,baseline_config.get("learning_rate",.01))
    solver = make_solver(config["physics"])
    misfit = MultiFrequencyMisfit(problem.observed,config["physics"]["dt"],**config["frequency"])
    history = []
    best_loss = float("inf")
    best_velocity = problem.initial_velocity.detach().clone()
    start_generation,prior_runtime = 0,0.
    stopping = Stopping(config.get("stopping",{}))
    reason = None

    def objective(velocity,environment):
        traces = solver(smooth_velocity(velocity,environment.model_smoothing_scale),problem.sources,problem.receivers,problem.wavelet)
        return misfit(traces,environment.frequency_weights,environment.center_frequency).mean()

    if state:
        optimizer.load_state_dict(state["optimizer"])
        for values in optimizer.state.values():
            for key,value in values.items():
                if isinstance(value,torch.Tensor):
                    values[key] = value.to(device)
        solver.counters = SolverCounters(**state["counters"])
        best_loss,initial_loss = state["best_loss"],state["initial_loss"]
        best_velocity = state["best_velocity"].to(device)
        history,start_generation,prior_runtime = state["history"],state["generation"],state["runtime"]
        stopping,reason = Stopping(**state["stopping"]),state["reason"]
        restore_rng(state["rng"])
    else:
        with torch.no_grad():
            initial_loss = float(objective(problem.initial_velocity,final_environment))
        best_loss = initial_loss
    start = time.perf_counter()
    end = config["generations"] if max_additional_generations is None else min(config["generations"],start_generation+max_additional_generations)
    for generation in range(start_generation,end):
        if reason:
            break
        environment = scheduler(generation)
        steps = baseline_config.get("steps_per_generation",environment.local_fwi_steps)
        for _ in range(steps):
            def closure():
                optimizer.zero_grad(set_to_none=True)
                value = objective(vmin+(vmax-vmin)*q.clamp(0,1),environment)
                if not torch.isfinite(value):
                    raise FloatingPointError("Nonfinite gradient baseline loss")
                value.backward()
                return value
            if optimizer_name.lower() == "lbfgs":
                optimizer.step(closure)
            else:
                closure()
                optimizer.step()
            with torch.no_grad():
                q.clamp_(0,1)
        with torch.no_grad():
            velocity = vmin+(vmax-vmin)*q
            current = float(objective(velocity,environment))
            reference = current if environment == final_environment else float(objective(velocity,final_environment))
            if reference < best_loss:
                best_loss,best_velocity = reference,velocity.detach().clone()
        reason = stopping.update(best_loss,progress=environment.progress) or ("max_generations" if generation+1==config["generations"] else None)
        history.append({"generation":generation,"best_fitness":current,"mean_fitness":current,"waveform_loss":current,
                        "reference_loss":best_loss,"frequency_weights":environment.frequency_weights,"mutation_sigma":0.,
                        "diversity":0.,"population_size":1,"bank_size":0,"local_fwi_steps":steps,
                        "pde_evaluations":solver.counters.model_evaluations})
        state = {"config":config,"environment_config":env_config,"problem":problem.state_dict(),"bounds":[vmin,vmax],
                 "q":q.detach().cpu(),"optimizer":optimizer.state_dict(),"counters":solver.counter_state(),"history":history,
                 "best_loss":best_loss,"initial_loss":initial_loss,"best_velocity":best_velocity.cpu(),
                 "generation":generation+1,"runtime":prior_runtime+time.perf_counter()-start,"stopping":vars(stopping),
                 "rng":rng_state(),"reason":reason}
        temporary = directory/"baseline.pt.tmp"
        torch.save(state,temporary)
        temporary.replace(directory/"baseline.pt")
        print(f"baseline generation={generation} reference={best_loss:.6g} PDE={solver.counters.model_evaluations}",flush=True)
    np.save(directory/"optimized_velocity.npy",best_velocity.cpu().numpy())
    best_velocity = smooth_velocity(best_velocity,final_environment.model_smoothing_scale).detach()
    np.save(directory/"recovered_velocity.npy",best_velocity.cpu().numpy())
    np.save(directory/"initial_velocity.npy",problem.initial_velocity.cpu().numpy())
    np.save(directory/"observed.npy",problem.observed.cpu().numpy())
    if problem.truth is not None:
        np.save(directory/"true_velocity.npy",problem.truth.cpu().numpy())
    summary = {"status":"complete" if reason else "checkpointed","stop_reason":reason,"generations_completed":len(history),
               "waveform_misfit":best_loss,"best_reference_fitness":best_loss,"initial_reference_fitness":initial_loss,
               "pde_evaluations":solver.counters.model_evaluations,**solver.counter_state(),
               "observation_pde_evaluations":problem.observation_pde_evaluations,"runtime_seconds":prior_runtime+time.perf_counter()-start,
               "qc_rejected":0,"injection_count":0,"successful_injection_count":0,"device":str(device),"output":str(directory),
               "velocity_bounds":[vmin,vmax],"metrics":velocity_metrics(best_velocity,problem.truth,vmax-vmin) if problem.truth is not None else {}}
    save_config(config,directory/"config.yaml")
    write_json(directory/"summary.json",summary)
    write_jsonl(directory/"history.jsonl",history)
    if plots:
        plot_results(directory/"figures",problem.initial_velocity,best_velocity,problem.truth,history)
    return summary
