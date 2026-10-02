#!/usr/bin/env python3
"""Publish sanitized ground-truth parquet pairs to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/tested-models/dataset-manifest.json"))
    parser.add_argument("--datasets-dir", type=Path, default=Path("data/tested-models/datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/tested-models/hf-ground-truth"))
    parser.add_argument("--repo-id", default="redhat-performance/configiq-performance-data")
    parser.add_argument("--revision", default=None, help="Optional branch or tag to update; defaults to the repository default branch.")
    return parser.parse_args()


def prepare(manifest_path: Path, datasets_dir: Path, output_dir: Path) -> dict:
    source = json.loads(manifest_path.read_text())
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    pairs = []
    for pair in source.get("pairs", []):
        pair_id = pair["id"]
        source_path = datasets_dir / f"{pair_id}.parquet"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination = output_dir / "pairs" / pair_id / "ground_truth.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        pairs.append({
            "id": pair_id,
            "model_id": pair["model_id"],
            "accelerator": pair["accelerator"],
            "records": pair.get("records"),
            "path": f"pairs/{pair_id}/ground_truth.parquet",
            "columns": pair.get("columns", []),
            "sha256": pair.get("sha256"),
        })
    manifest = {
        "schema_version": 1,
        "source": "configiq-tested-models-ground-truth",
        "pairs": pairs,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    args = parse_args()
    manifest = prepare(args.manifest, args.datasets_dir, args.output_dir)
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(args.output_dir),
        revision=args.revision,
        commit_message=f"Publish {len(manifest['pairs'])} ground-truth model hardware pairs",
    )
    print(f"published {len(manifest['pairs'])} pairs to {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
