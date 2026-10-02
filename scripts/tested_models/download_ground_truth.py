#!/usr/bin/env python3
"""Download a pinned public ground-truth dataset into the local pipeline layout."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="redhat-performance/configiq-performance-data")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/tested-models"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise SystemExit("--revision must be a full 40-character lowercase Hugging Face commit SHA")
    snapshot = Path(snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=["manifest.json", "pairs/*/ground_truth.parquet"],
    ))
    source_manifest = json.loads((snapshot / "manifest.json").read_text())
    datasets_dir = args.output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for pair in source_manifest.get("pairs", []):
        pair_id = pair.get("id")
        if not isinstance(pair_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", pair_id):
            raise ValueError(f"invalid filesystem pair id: {pair_id!r}")
        pair_path = Path(pair["path"])
        if pair_path.is_absolute() or ".." in pair_path.parts:
            raise ValueError(f"invalid dataset path: {pair_path}")
        source_path = snapshot / pair_path
        destination = datasets_dir / f"{pair_id}.parquet"
        shutil.copy2(source_path, destination)
        pairs.append({**pair, "path": str(destination)})
    manifest = {"schema_version": 1, "source": args.repo_id, "revision": args.revision, "pairs": pairs}
    (args.output_dir / "dataset-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"downloaded {len(pairs)} ground-truth pairs from {args.repo_id}@{args.revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
