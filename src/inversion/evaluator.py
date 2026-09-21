"""Connect decoder, QC, one wave solve, multiband loss and fitness."""
from dataclasses import dataclass, replace
import math
import torch
import torch.nn.functional as F
from src.qc.quality_control import QualityControl


def smooth_velocity(velocity: torch.Tensor, sigma: float) -> torch.Tensor:
    """Differentiable Gaussian smoothing with replicated physical boundaries."""
    if sigma < 1e-4:
        return velocity
    radius = max(1,math.ceil(3*sigma))
    axis = torch.arange(-radius,radius+1,device=velocity.device,dtype=velocity.dtype)
    kernel = torch.exp(-axis.square()/(2*sigma*sigma))
    kernel = kernel/kernel.sum()
    weight = (kernel[:,None]*kernel[None,:])[None,None]
    return F.conv2d(F.pad(velocity,(radius,radius,radius,radius),mode="replicate"),weight)


@dataclass
class Evaluation:
    """One gene's objective components, all tied to the supplied environment."""
    total: torch.Tensor
    waveform: torch.Tensor
    cycle: torch.Tensor
    geological: torch.Tensor
    band_losses: torch.Tensor
    accepted: bool
    reason: str

    def detached(self):
        return replace(self,total=self.total.detach(),waveform=self.waveform.detach(),cycle=self.cycle.detach(),
                       geological=self.geological.detach(),band_losses=self.band_losses.detach())

    def scalars(self) -> dict:
        return {"fitness":float(self.total.detach()),"waveform_loss":float(self.waveform.detach()),
                "cycle_loss":float(self.cycle.detach()),"geological_score":float(self.geological.detach()),
                "band_losses":self.band_losses.detach().cpu().tolist(),"accepted":self.accepted,"reason":self.reason}


class Evaluator:
    """Never uses ground-truth velocity for optimization or QC."""
    def __init__(self, autoencoder, solver, misfit, sources: torch.Tensor,
                 receivers: torch.Tensor, wavelet: torch.Tensor, config: dict, shape: tuple[int,int]) -> None:
        self.autoencoder,self.solver,self.misfit = autoencoder,solver,misfit
        self.sources,self.receivers,self.wavelet = sources,receivers,wavelet
        self.qc = QualityControl(autoencoder,config,shape)
        self.evaluations = 0
        self.rejected = 0
        self.rejections: dict[str,int] = {}

    def __call__(self, z: torch.Tensor, environment) -> Evaluation:
        self.evaluations += 1
        result = self.qc(z,environment)
        if not result.passed:
            self.rejected += 1
            self.rejections[result.reason] = self.rejections.get(result.reason,0)+1
            inf = z.new_tensor(float("inf"))
            return Evaluation(inf,inf,result.cycle_loss,result.geological_penalty,z.new_full((3,),float("inf")),False,result.reason)
        velocity = smooth_velocity(result.velocity,environment.model_smoothing_scale)
        predicted = self.solver(velocity,self.sources,self.receivers,self.wavelet)
        components = self.misfit.components(predicted,environment.center_frequency)[0]
        waveform = (components*z.new_tensor(environment.frequency_weights)).sum()
        total = environment.lambda_wave*waveform + environment.lambda_cycle*result.cycle_loss + environment.lambda_geo*result.geological_penalty
        return Evaluation(total,waveform,result.cycle_loss,result.geological_penalty,components,True,"accepted")

    def population(self, genes: torch.Tensor, environment) -> list[Evaluation]:
        with torch.no_grad():
            return [self(gene,environment) for gene in genes]
