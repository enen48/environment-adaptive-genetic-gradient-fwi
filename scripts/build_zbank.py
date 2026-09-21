"""Initialize bounded memory from training velocity models only."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from src.zbank.initializer import training_latents,make_bank,model_signature
from src.models.autoencoder import load_autoencoder
from src.environment.scheduler import EnvironmentScheduler
from src.zbank.bank import ZBank
from src.zbank.entry import BankEntry
from src.utils.config import load_config,storage_path
from src.utils.seed import setup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",default="checkpoints/autoencoder.pt")
    parser.add_argument("--output",default="checkpoints/zbank.pt")
    parser.add_argument("--environment",default="configs/environment.yaml")
    parser.add_argument("--capacity",type=int,default=128)
    args = parser.parse_args()
    device = setup(42)
    model,config = load_autoencoder(storage_path(args.checkpoint),device)
    model.freeze()
    genes = training_latents(model,config,args.capacity)
    environment = EnvironmentScheduler(load_config(args.environment),30)(0)
    bank = make_bank(genes,environment,args.capacity)
    bank.model_signature = model_signature(model)
    bank.save(storage_path(args.output))
    print(f"Saved {len(bank)} training-only latent entries to {storage_path(args.output)}")


if __name__ == "__main__":
    main()
