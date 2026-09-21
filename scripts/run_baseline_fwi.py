"""Pure gradient multiscale acoustic FWI baseline."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.fwi.baseline import run_baseline
from src.utils.config import load_config,storage_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/inversion.yaml")
    parser.add_argument("--output",default=None)
    parser.add_argument("--resume",default=None)
    parser.add_argument("--max-additional-generations",type=int,default=None)
    parser.add_argument("--no-plots",action="store_true")
    args = parser.parse_args()
    result = run_baseline(load_config(args.config),args.output,not args.no_plots,
                          storage_path(args.resume) if args.resume else None,args.max_additional_generations)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
