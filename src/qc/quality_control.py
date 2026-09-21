"""One decoder pass shared by cycle QC, geological QC and candidate evaluation."""
from dataclasses import dataclass
import torch
from .latent_qc import LatentQC
from .geological_qc import GeologicalQC


@dataclass
class QCResult:
    """QC outputs carry differentiable penalties for local refinement."""
    velocity: torch.Tensor | None
    cycle_loss: torch.Tensor
    geological_penalty: torch.Tensor
    passed: bool
    reason: str = "accepted"


class QualityControl:
    """Cycle-first/geo-second filtering; rejected genes never reach the solver."""
    def __init__(self, autoencoder, config: dict, shape: tuple[int,int]) -> None:
        self.autoencoder,self.config,self.shape = autoencoder,config,shape
        self.enabled = config.get("enabled",True)
        self.latent = LatentQC(config)
        self.geological = GeologicalQC(config)

    def __call__(self, z: torch.Tensor, environment) -> QCResult:
        zero = z.new_zeros(())
        if not torch.isfinite(z).all():
            return QCResult(None,z.new_tensor(float("inf")),zero,False,"nonfinite_latent")
        model = self.autoencoder.decode(z.reshape(1,-1),self.shape)
        reconstructed = self.autoencoder.encode(model)
        cycle = (z.reshape_as(reconstructed)-reconstructed).square().sum()
        if self.enabled and (not torch.isfinite(cycle) or float(cycle.detach().sqrt()) > self.latent.limit(environment.qc_threshold)):
            return QCResult(model,cycle,zero,False,"latent_cycle")
        geological = self.geological(model)
        passed = geological.passed if self.enabled else bool(torch.isfinite(model).all())
        return QCResult(model,cycle,geological.penalty,passed,"accepted" if passed else "geological")

    def calibrate(self, genes: torch.Tensor) -> None:
        with torch.no_grad():
            model = self.autoencoder.decode(genes,self.shape)
            norms = (genes-self.autoencoder.encode(model)).norm(dim=-1)
        self.latent.calibrate(norms)
