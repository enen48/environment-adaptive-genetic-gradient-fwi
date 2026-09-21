"""Small explicit optimizer factory; unknown names fail early."""
import torch


def make_optimizer(parameters, name: str = "Adam", learning_rate: float = .03):
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if name.lower() == "adam":
        return torch.optim.Adam(parameters,lr=learning_rate)
    if name.lower() == "sgd":
        return torch.optim.SGD(parameters,lr=learning_rate)
    if name.lower() == "lbfgs":
        return torch.optim.LBFGS(parameters,lr=learning_rate,max_iter=1,max_eval=5,history_size=10,
                                 tolerance_grad=1e-10,tolerance_change=1e-12,line_search_fn="strong_wolfe")
    raise ValueError(f"Unknown local optimizer: {name}")
