#!/usr/bin/env python3
"""Publish sanitized ground-truth parquet pairs to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import json
import re
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
    manifest_path = manifest_path.resolve()
    datasets_dir = datasets_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == manifest_path or output_dir == datasets_dir:
        raise ValueError("output directory must not equal an input path")
    if output_dir.is_relative_to(manifest_path) or output_dir.is_relative_to(datasets_dir):
        raise ValueError("output directory must not be inside an input path")
    if manifest_path.is_relative_to(output_dir) or datasets_dir.is_relative_to(output_dir):
        raise ValueError("output directory must not contain an input path")
    source = json.loads(manifest_path.read_text())
    pairs = source.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("ground-truth manifest must contain at least one pair")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    pairs = []
    for pair in pairs:
        pair_id = pair["id"]
        if not isinstance(pair_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", pair_id):
            raise ValueError(f"invalid filesystem pair id: {pair_id!r}")
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
        delete_patterns=["pairs/**", "manifest.json"],
        commit_message=f"Publish {len(manifest['pairs'])} ground-truth model hardware pairs",
    )
    print(f"published {len(manifest['pairs'])} pairs to {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
