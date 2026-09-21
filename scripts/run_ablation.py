"""Run repeated-seed baselines/ablations and aggregate actual measured costs."""
import argparse
import csv
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.inversion.ablation import variant_config
from src.inversion.engine import InversionEngine
from src.fwi.baseline import run_baseline
from src.utils.config import load_config,storage_path
from src.utils.logger import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/experiment.yaml")
    parser.add_argument("--seeds",type=int,nargs="+",default=None)
    parser.add_argument("--variants",nargs="+",default=None)
    parser.add_argument("--no-plots",action="store_true")
    args = parser.parse_args()
    experiment = load_config(args.config)
    original = load_config(experiment["inversion_config"])
    directory = storage_path(experiment["output"])
    directory.mkdir(parents=True,exist_ok=True)
    rows = []
    for seed in args.seeds or experiment["seeds"]:
        for variant in args.variants or experiment["variants"]:
            config = dict(original,seed=seed,output=str(directory/f"{variant}_seed{seed}"))
            if variant == "gradient":
                summary = run_baseline(config,config["output"],not args.no_plots)
            else:
                config = variant_config(config,variant)
                summary = InversionEngine(config).run(plots=not args.no_plots)
            rows.append({"variant":variant,"seed":seed,"waveform_misfit":summary["waveform_misfit"],
                "pde_evaluations":summary["pde_evaluations"],"runtime_seconds":summary["runtime_seconds"],
                "injections":summary["injection_count"],"successful_injections":summary["successful_injection_count"],
                "qc_rejected":summary["qc_rejected"],"stop_reason":summary["stop_reason"],**summary["metrics"]})
            write_json(directory/"results.json",rows)
    with (directory/"results.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    aggregates = []
    for variant in dict.fromkeys(row["variant"] for row in rows):
        selected = [row for row in rows if row["variant"]==variant]
        aggregate = {"variant":variant,"n":len(selected)}
        for key in ["waveform_misfit","velocity_rmse","pde_evaluations","runtime_seconds"]:
            values = [row[key] for row in selected if key in row]
            if values:
                aggregate[key+"_mean"] = float(np.mean(values))
                aggregate[key+"_std"] = float(np.std(values,ddof=1)) if len(values)>1 else None
        aggregates.append(aggregate)
    write_json(directory/"aggregates.json",aggregates)
    print(f"Saved {len(rows)} measured runs to {directory}")


if __name__ == "__main__":
    main()
