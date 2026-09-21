"""Verify bundled data or fetch pinned full velocity shards, never seismic gathers."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import requests
from src.utils.config import ROOT, storage_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-shards", action="store_true", help="Download two 9.8 MB velocity shards with SHA256 verification")
    parser.add_argument("--output", type=Path, default=storage_path("data/openfwi"))
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/sample/provenance.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        bundled = ROOT / item["path"]
        if sha256(bundled) != item["sha256"]:
            raise ValueError(f"Bundled data checksum mismatch: {bundled}")
        arr = np.load(bundled, mmap_mode="r", allow_pickle=False)
        if list(arr.shape) != item["shape"] or arr.dtype != np.float32:
            raise ValueError(f"Unexpected data shape/dtype: {bundled}")
        print(f"Verified {bundled.name}: {arr.shape}, float32, m/s")
        if args.full_shards:
            args.output.mkdir(parents=True, exist_ok=True)
            target = args.output / Path(item["source_shard"]).name
            if target.exists() and sha256(target) == item["source_sha256"]:
                print(f"Verified existing shard: {target}")
                continue
            temporary = target.with_suffix(".npy.tmp")
            with requests.get(item["source_url"], stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                with temporary.open("wb") as out:
                    for block in response.iter_content(1024 * 1024):
                        out.write(block)
            if temporary.stat().st_size != item["source_size_bytes"] or sha256(temporary) != item["source_sha256"]:
                raise ValueError(f"Downloaded shard verification failed; kept {temporary} for inspection")
            temporary.replace(target)
            print(f"Verified downloaded shard: {target}")


if __name__ == "__main__":
    main()
