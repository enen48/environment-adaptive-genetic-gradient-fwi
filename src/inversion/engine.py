"""Resumable GA/QC/physics/local-FWI/memory loop with explicit cost accounting."""
from __future__ import annotations
import copy
import math
import time
from pathlib import Path
import numpy as np
import torch
from src.models.autoencoder import AutoEncoder,load_autoencoder
from src.environment.scheduler import EnvironmentScheduler
from src.ga.genetic_optimizer import GeneticOptimizer
from src.ga.diversity import diversity
from src.zbank.bank import ZBank
from src.zbank.entry import BankEntry
from src.zbank.injection import inject
from src.fwi.local_refinement import refine_population
from src.physics.misfit import MultiFrequencyMisfit
from src.physics.wave_solver import SolverCounters
from src.inversion.evaluator import Evaluator,smooth_velocity
from src.inversion.problem import Problem,make_problem,make_solver
from src.inversion.stopping import Stopping
from src.utils.config import load_config,merge,storage_path,save_config
from src.utils.seed import setup,rng_state,restore_rng
from src.utils.logger import write_json,write_jsonl
from src.utils.metrics import velocity_metrics
from src.utils.visualization import plot_results
from src.zbank.initializer import training_latents,make_bank,model_signature


class InversionEngine:
    """Optimization does not inspect true velocity; truth is used for reporting only.

    Checkpoints are written at generation boundaries with the NEXT population,
    complete RNG state and the original schedule horizon. The fixed final
    environment reference objective governs best-model and stopping decisions.
    """
    def __init__(self, config: dict, resume: str | Path | None = None) -> None:
        self.config = copy.deepcopy(config)
        self.device = setup(config.get("seed",43),config.get("device","auto"),config.get("threads",2))
        self.output = storage_path(config["output"])
        self.output.mkdir(parents=True,exist_ok=True)
        state = torch.load(resume,map_location="cpu",weights_only=True) if resume else None
        env_config = merge(load_config(config["environment_config"]),config.get("environment_overrides",{}))
        if state:
            if state.get("format_version") != 2:
                raise ValueError("Unsupported inversion checkpoint")
            old,new = copy.deepcopy(state["config"]),copy.deepcopy(config)
            old.pop("output",None)
            new.pop("output",None)
            if old != new or env_config != state["environment_config"]:
                raise ValueError("Resume config differs; preserve schedule horizon/data/options (output may change)")
            self.model = AutoEncoder(**state["model_config"]).to(self.device)
            self.model.load_state_dict(state["model_state"])
            self.training_config = state["training_config"]
            self.problem = Problem.from_state_dict(state["problem"],self.device)
            self.genes = state["genes"].to(self.device)
            self.prior = state["prior"].to(self.device)
            self.bank = ZBank.from_state_dict(state["bank"])
        else:
            checkpoint = storage_path(config["checkpoint"])
            if not checkpoint.exists():
                raise FileNotFoundError(f"Train the AE first: missing {checkpoint}")
            self.model,self.training_config = load_autoencoder(checkpoint,self.device)
            self.model.freeze()
            self.prior = training_latents(self.model,self.training_config,max(128,config["zbank"].get("capacity",128)))
        self.model.freeze()
        self.environment_config = env_config
        self.scheduler = EnvironmentScheduler(env_config,config["generations"],config.get("dynamic_environment",True))
        self.reference_environment = self.scheduler.at_progress(1.)
        self.memory_enabled = config["zbank"].get("enabled",True)
        if not state:
            initial_environment = self.scheduler(0)
            bank_path = storage_path(config.get("bank","checkpoints/zbank.pt"))
            loading_bank = self.memory_enabled and bank_path.exists()
            self.bank = ZBank.load(bank_path) if loading_bank else make_bank(self.prior,initial_environment,config["zbank"].get("capacity",128))
            signature = model_signature(self.model)
            if loading_bank and self.bank.model_signature != signature:
                raise ValueError("Z-bank was built for different/unknown AE weights; rerun scripts/build_zbank.py for this checkpoint")
            self.bank.model_signature = signature
            if self.bank.entries and self.bank.entries[0].z.numel() != self.model.latent_dim:
                raise ValueError("Z-bank latent dimension differs from AE; rebuild bank")
            self.bank.capacity = config["zbank"].get("capacity",128)
            self.bank.prune()
            if not self.memory_enabled:
                self.bank = ZBank(config["zbank"].get("capacity",128))
            n = initial_environment.population_target_size
            mean = self.prior.mean(0)
            scale = self.prior.std(0,unbiased=False).clamp_min(.05)
            self.genes = mean + torch.randn(n,self.model.latent_dim,device=self.device)*scale
            # Half starts from empirical prior genes; first always shares baseline initial model.
            selected = torch.randperm(len(self.prior),device=self.device)[:min(n//2,len(self.prior))]
            self.genes[:len(selected)] = self.prior[selected]
            self.genes[0] = mean
            if self.memory_enabled:
                remembered = self.bank.query(initial_environment.vector(),self.genes,min(n//4,len(self.bank)),allow_unknown=True)
                for i,entry in enumerate(remembered):
                    self.genes[-1-i] = entry.z.to(self.genes)
            with torch.no_grad():
                initial = self.model.decode(mean[None],tuple(config["shape"]))
            self.problem = make_problem(config,initial,self.device)
        solver = make_solver(config["physics"])
        misfit = MultiFrequencyMisfit(self.problem.observed,config["physics"]["dt"],**config["frequency"])
        self.evaluator = Evaluator(self.model,solver,misfit,self.problem.sources,self.problem.receivers,
                                   self.problem.wavelet,config["qc"],tuple(config["shape"]))
        self.evaluator.qc.calibrate(self.prior)
        self.ga = GeneticOptimizer(config["ga"])
        self.stop = Stopping(config.get("stopping",{}))
        self.generation = 0
        self.history: list[dict] = []
        self.best_z: torch.Tensor | None = None
        self.best_reference = float("inf")
        self.best_reference_waveform = float("inf")
        self.initial_reference = float("inf")
        self.injection_count = 0
        self.accepted_injections = 0
        self.successful_injections = 0
        self.pending_injections: list[int] = []
        self.runtime = 0.
        self.stop_reason = None
        if state:
            for key in ["generation","history","best_reference","best_reference_waveform","initial_reference",
                        "injection_count","accepted_injections","successful_injections","pending_injections","runtime","stop_reason"]:
                setattr(self,key,state[key])
            self.best_z = state["best_z"].to(self.device) if state["best_z"] is not None else None
            self.evaluator.solver.counters = SolverCounters(**state["solver_counters"])
            self.evaluator.evaluations = state["candidate_evaluations"]
            self.evaluator.rejected = state["rejected"]
            self.evaluator.rejections = state["rejections"]
            vars(self.evaluator.qc.latent).update(state["qc_state"])
            self.stop = Stopping(**state["stopping_state"])
            restore_rng(state["rng"])
        else:
            with torch.no_grad():
                initial_result = self.evaluator(self.prior.mean(0),self.reference_environment)
            if initial_result.accepted:
                self._consider_reference(self.prior.mean(0),initial_result)
                self.initial_reference = float(initial_result.total)
        save_config(config,self.output/"config.yaml")
        save_config(env_config,self.output/"environment.yaml")

    def _consider_reference(self,z: torch.Tensor,result) -> None:
        if result.accepted and float(result.total) < self.best_reference:
            self.best_reference = float(result.total)
            self.best_reference_waveform = float(result.waveform)
            self.best_z = z.detach().clone()

    def save(self) -> Path:
        """Atomic full snapshot using only safe primitives and tensors."""
        state = {"format_version":2,"config":self.config,"environment_config":self.environment_config,
                 "model_config":self.model.architecture_config(),"model_state":{k:v.detach().cpu() for k,v in self.model.state_dict().items()},
                 "training_config":self.training_config,"problem":self.problem.state_dict(),
                 "genes":self.genes.detach().cpu(),"prior":self.prior.detach().cpu(),"bank":self.bank.state_dict(),
                 "best_z":self.best_z.detach().cpu() if self.best_z is not None else None,
                 "solver_counters":self.evaluator.solver.counter_state(),"candidate_evaluations":self.evaluator.evaluations,
                 "rejected":self.evaluator.rejected,"rejections":self.evaluator.rejections,
                 "qc_state":self.evaluator.qc.latent.state_dict(),"stopping_state":vars(self.stop),"rng":rng_state()}
        for key in ["generation","history","best_reference","best_reference_waveform","initial_reference",
                    "injection_count","accepted_injections","successful_injections","pending_injections","runtime","stop_reason"]:
            state[key] = getattr(self,key)
        checkpoint = self.output/"inversion.pt"
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save(state,temporary)
        temporary.replace(checkpoint)
        return checkpoint

    def run(self, max_additional_generations: int | None = None, plots: bool = True) -> dict:
        start = time.perf_counter()
        before = self.runtime
        end = self.config["generations"]
        if max_additional_generations is not None:
            end = min(end,self.generation+max_additional_generations)
        for generation in range(self.generation,end):
            if self.stop_reason:
                break
            environment = self.scheduler(generation)
            results = self.evaluator.population(self.genes,environment)
            scores = torch.tensor([float(r.total) for r in results],device=self.device)
            finite = torch.isfinite(scores)
            if not finite.any():
                self.stop_reason = "no_feasible_candidates"
                self.runtime = before+time.perf_counter()-start
                self.save()
                break
            injection_success = None
            successful = 0
            accepted_injections = sum(results[i].accepted for i in self.pending_injections)
            self.accepted_injections += accepted_injections
            self.genes,results,local_improvements,refined_count = refine_population(self.genes,results,self.evaluator,environment,self.config["local_fwi"])
            scores = torch.tensor([float(r.total) for r in results],device=self.device)
            if self.pending_injections:
                remaining = [i for i in range(len(scores)) if i not in self.pending_injections]
                incumbent = float(scores[remaining].min()) if remaining else float("inf")
                successful = sum(float(scores[i]) < incumbent for i in self.pending_injections)
                self.successful_injections += successful
                injection_success = successful > 0
                self.pending_injections = []
            finite = torch.isfinite(scores)
            ranked = scores.argsort()
            best_index = int(ranked[0])
            accepted_norms = torch.stack([result.cycle.sqrt() for result in results if result.accepted])
            self.evaluator.qc.latent.update(accepted_norms)
            if self.best_z is not None:
                with torch.no_grad():
                    incumbent_feasible = self.evaluator.qc(self.best_z,self.reference_environment).passed
                if not incumbent_feasible:
                    self.best_z,self.best_reference,self.best_reference_waveform = None,float("inf"),float("inf")
                    self.stop.best,self.stop.stale = float("inf"),0
            # Fixed monitoring incurs an extra forward when objective/smoothing differ.
            reference_budget = self.config.get("reference_top_k")
            reference_count = int(finite.sum()) if reference_budget is None else min(reference_budget,int(finite.sum()))
            references = {}
            with torch.no_grad():
                for index in ranked[:reference_count].tolist():
                    reference = results[index] if environment == self.reference_environment else self.evaluator(self.genes[index],self.reference_environment)
                    references[index] = reference
                    self._consider_reference(self.genes[index],reference)
            population_diversity = diversity(self.genes,**self.config.get("diversity",{}))
            if self.memory_enabled:
                for index in ranked[:min(3,int(finite.sum()))].tolist():
                    result = results[index]
                    reference = references.get(index)
                    canonical = float(reference.total) if reference is not None and reference.accepted else None
                    self.bank.add(BankEntry(self.genes[index],environment.vector(),float(result.total),float(result.waveform),
                        float(result.cycle),float(result.geological),population_diversity,generation,environment.frequency_weights,
                        reference_fitness=canonical))
            reason = self.stop.update(self.best_reference,injection_success,progress=environment.progress)
            next_genes = self.genes
            injected = 0
            if generation+1 < self.config["generations"] and reason is None:
                next_environment = self.scheduler(generation+1)
                # Current operator parameters, next generation's target size.
                from dataclasses import replace
                next_genes = self.ga.step(self.genes,scores,replace(environment,population_target_size=next_environment.population_target_size))
                next_diversity = diversity(next_genes,**self.config.get("diversity",{}))
                if self.memory_enabled and next_diversity < next_environment.diversity_threshold:
                    bank_config = self.config["zbank"]
                    next_genes,indices = inject(next_genes,self.bank,next_environment.vector(),next_environment.injection_count,
                        bank_config.get("max_injection",3),protected=max(1,math.ceil(len(next_genes)*next_environment.elite_ratio)),
                        alpha=bank_config.get("fitness_weight",1.),beta=bank_config.get("diversity_weight",1.),gamma=bank_config.get("environment_weight",.5))
                    injected = len(indices)
                    self.injection_count += injected
                    self.pending_injections = indices
            row = {"generation":generation,"best_fitness":float(scores[best_index]),"mean_fitness":float(scores[finite].mean()),
                   "waveform_loss":float(results[best_index].waveform),"band_losses":results[best_index].band_losses.cpu().tolist(),
                   "reference_loss":self.best_reference,"diversity":population_diversity,"population_size":len(self.genes),
                   "bank_size":len(self.bank),"mutation_sigma":environment.mutation_sigma,"frequency_weights":environment.frequency_weights,
                   "local_fwi_steps":environment.local_fwi_steps,"refined_count":refined_count,"local_improvements":local_improvements,
                   "pde_evaluations":self.evaluator.solver.counters.model_evaluations,"solver_calls":self.evaluator.solver.counters.calls,
                   "shot_evaluations":self.evaluator.solver.counters.shot_evaluations,"rejected_genes":self.evaluator.rejected,
                   "injected":injected,"accepted_injections":accepted_injections,"successful_injections":successful,
                   "reference_evaluated_candidates":reference_count,"environment":environment.state_dict()}
            self.history.append(row)
            self.genes = next_genes.detach()
            self.generation = generation+1
            self.stop_reason = reason or ("max_generations" if self.generation >= self.config["generations"] else None)
            self.runtime = before+time.perf_counter()-start
            write_jsonl(self.output/"history.jsonl",self.history)
            self.save()
            print(f"generation={generation} fitness={row['best_fitness']:.6g} reference={self.best_reference:.6g} PDE={row['pde_evaluations']} QC_rejected={row['rejected_genes']}",flush=True)
        self.runtime = before+time.perf_counter()-start
        if self.best_z is None:
            raise RuntimeError("No candidate satisfies final-environment QC; inspect thresholds, AE training and history")
        return self.report(plots)

    def report(self,plots: bool = True) -> dict:
        with torch.no_grad():
            decoded = self.model.decode(self.best_z[None],tuple(self.config["shape"]))
            recovered = smooth_velocity(decoded,self.reference_environment.model_smoothing_scale)
        np.save(self.output/"decoded_velocity.npy",decoded.cpu().numpy())
        np.save(self.output/"recovered_velocity.npy",recovered.cpu().numpy())
        np.save(self.output/"initial_velocity.npy",self.problem.initial_velocity.cpu().numpy())
        np.save(self.output/"observed.npy",self.problem.observed.cpu().numpy())
        np.save(self.output/"best_z.npy",self.best_z.cpu().numpy())
        if self.problem.truth is not None:
            np.save(self.output/"true_velocity.npy",self.problem.truth.cpu().numpy())
        metrics = velocity_metrics(recovered,self.problem.truth,self.model.velocity_max-self.model.velocity_min) if self.problem.truth is not None else {}
        summary = {"status":("failed" if self.stop_reason == "no_feasible_candidates" else "complete") if self.stop_reason else "checkpointed","stop_reason":self.stop_reason,
                   "generations_completed":self.generation,"best_reference_fitness":self.best_reference,
                   "initial_reference_fitness":self.initial_reference,"waveform_misfit":self.best_reference_waveform,
                   "pde_evaluations":self.evaluator.solver.counters.model_evaluations,**self.evaluator.solver.counter_state(),
                   "observation_pde_evaluations":self.problem.observation_pde_evaluations,"qc_rejected":self.evaluator.rejected,
                   "qc_rejection_reasons":self.evaluator.rejections,"injection_count":self.injection_count,
                   "accepted_injection_count":self.accepted_injections,"successful_injection_count":self.successful_injections,"runtime_seconds":self.runtime,
                   "device":str(self.device),"velocity_bounds":[self.model.velocity_min,self.model.velocity_max],"metrics":metrics,"output":str(self.output)}
        write_json(self.output/"summary.json",summary)
        if plots:
            plot_results(self.output/"figures",self.problem.initial_velocity,recovered,self.problem.truth,self.history)
        return summary
