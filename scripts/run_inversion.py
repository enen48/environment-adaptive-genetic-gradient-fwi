"""Run/resume the online latent genetic-gradient inversion."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.inversion.engine import InversionEngine
from src.utils.config import load_config,storage_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/inversion.yaml")
    parser.add_argument("--resume",default=None)
    parser.add_argument("--max-additional-generations",type=int,default=None,help="Checkpoint early without changing schedule horizon")
    parser.add_argument("--no-plots",action="store_true")
    args = parser.parse_args()
    engine = InversionEngine(load_config(args.config),storage_path(args.resume) if args.resume else None)
    print(json.dumps(engine.run(args.max_additional_generations,not args.no_plots),indent=2))


if __name__ == "__main__":
    main()
