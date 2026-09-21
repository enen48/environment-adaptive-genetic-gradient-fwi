"""Seed all RNGs used by the project."""
import random
import numpy as np
import torch


def setup(seed: int, device: str = "auto", threads: int = 2) -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)


def rng_state() -> dict:
    """Serializable RNG state for exact continuation on the same backend."""
    np_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [np_state[0],np_state[1].tolist(),np_state[2],np_state[3],np_state[4]], "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0],np.array(np_state[1],dtype=np.uint32),np_state[2],np_state[3],np_state[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
