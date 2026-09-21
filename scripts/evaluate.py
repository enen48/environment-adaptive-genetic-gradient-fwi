"""Recompute physical velocity metrics and optionally redraw stored run figures."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.utils.config import storage_path
from src.utils.metrics import velocity_metrics
from src.utils.visualization import plot_results
from src.utils.logger import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default="outputs/inversion")
    parser.add_argument("--plots",action="store_true")
    args = parser.parse_args()
    directory = storage_path(args.run)
    recovered = torch.from_numpy(np.load(directory/"recovered_velocity.npy",allow_pickle=False))
    initial = torch.from_numpy(np.load(directory/"initial_velocity.npy",allow_pickle=False))
    truth_path = directory/"true_velocity.npy"
    truth = torch.from_numpy(np.load(truth_path,allow_pickle=False)) if truth_path.exists() else None
    result = {"recovered":velocity_metrics(recovered,truth),"initial":velocity_metrics(initial,truth)} if truth is not None else {"message":"No true velocity supplied; only waveform metrics are available in summary.json"}
    write_json(directory/"evaluation.json",result)
    if args.plots:
        history = [json.loads(line) for line in (directory/"history.jsonl").read_text().splitlines()]
        plot_results(directory/"figures",initial,recovered,truth,history)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
