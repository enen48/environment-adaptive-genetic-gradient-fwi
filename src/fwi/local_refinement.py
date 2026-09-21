"""Top-K local search: ∇z J(D(z))=(∂D/∂z)^T ∇m J."""
from __future__ import annotations
import torch
from .optimizer import make_optimizer


class RejectedStep(RuntimeError):
    """A trial left the feasible latent/model region."""


def refine_gene(gene: torch.Tensor, initial, evaluator, environment, steps: int,
                optimizer: str = "Adam", learning_rate: float = .03,
                latent_clip: float | None = 6.) -> tuple[torch.Tensor,object]:
    """Return the best feasible iterate; worsening/nonfinite updates are discarded."""
    if not initial.accepted or steps <= 0:
        return gene.detach().clone(),initial
    z = gene.detach().clone().requires_grad_()
    method = make_optimizer([z],optimizer,learning_rate)
    best_z,best_result = z.detach().clone(),initial.detached()

    def closure():
        nonlocal best_z,best_result
        method.zero_grad(set_to_none=True)
        result = evaluator(z,environment)
        if not result.accepted or not torch.isfinite(result.total):
            raise RejectedStep("Local trial failed QC")
        if result.total.detach() < best_result.total:
            best_z,best_result = z.detach().clone(),result.detached()
        result.total.backward()
        if z.grad is None or not torch.isfinite(z.grad).all():
            raise RejectedStep("Nonfinite latent gradient")
        torch.nn.utils.clip_grad_norm_([z],max_norm=10.)
        return result.total

    for _ in range(steps):
        try:
            if optimizer.lower() == "lbfgs":
                method.step(closure)
            else:
                closure()
                method.step()
            with torch.no_grad():
                if latent_clip is not None:
                    z.clamp_(-latent_clip,latent_clip)
        except RejectedStep:
            break
    with torch.no_grad():
        final = evaluator(z,environment)
        if final.accepted and torch.isfinite(final.total) and final.total < best_result.total:
            best_z,best_result = z.detach().clone(),final.detached()
    return best_z,best_result


def refine_population(genes: torch.Tensor, results: list, evaluator, environment,
                      config: dict) -> tuple[torch.Tensor,list,int,int]:
    """Strictly fewer than all population members can receive local FWI."""
    if not config.get("enabled",True):
        return genes,results,0,0
    scores = torch.tensor([float(r.total.detach()) for r in results],device=genes.device)
    valid = torch.where(torch.isfinite(scores))[0]
    count = min(max(0,config.get("top_k",3)),max(0,len(genes)-1),len(valid))
    output,updated = genes.clone(),list(results)
    improved = 0
    for index in valid[scores[valid].argsort()[:count]].tolist():
        z,result = refine_gene(genes[index],results[index],evaluator,environment,environment.local_fwi_steps,
                               config.get("optimizer","Adam"),config.get("learning_rate",.03),config.get("latent_clip",6.))
        if result.total < results[index].total:
            improved += 1
            output[index],updated[index] = z,result
    return output.detach(),updated,improved,count
