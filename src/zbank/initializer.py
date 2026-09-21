"""Training-only latent prior and offline memory initialization."""
from __future__ import annotations
import hashlib
import json
import torch
from torch.utils.data import DataLoader
from src.data.loaders import make_datasets
from src.zbank.bank import ZBank
from src.zbank.entry import BankEntry


def model_signature(model) -> str:
    digest = hashlib.sha256(json.dumps(model.architecture_config(),sort_keys=True).encode())
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def training_latents(model,config: dict,limit: int = 128) -> torch.Tensor:
    """Use the exact training partition; never sample validation or observed data."""
    train,_ = make_datasets(config)
    device = next(model.parameters()).device
    encoded = []
    with torch.no_grad():
        for batch in DataLoader(train,batch_size=16,shuffle=False,num_workers=0):
            encoded.append(model.encode(batch.to(device)))
            if sum(len(value) for value in encoded) >= limit:
                break
    return torch.cat(encoded)[:limit].detach()


def make_bank(genes: torch.Tensor,environment,capacity: int = 128) -> ZBank:
    bank = ZBank(capacity)
    for z in genes:
        bank.add(BankEntry(z,environment.vector(),float("inf"),frequency_weights=environment.frequency_weights))
    return bank


