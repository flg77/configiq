#!/usr/bin/env python3
"""Export AISimulate's packaged HF configs into a BLIS catalog overlay."""

from __future__ import annotations

import json
import re
import shutil
import sys
from importlib import resources
from pathlib import Path


def model_id_from_filename(path: Path) -> str:
    stem = path.name.removesuffix("_config.json")
    return stem.replace("--", "/", 1)


def safe_name(model_id: str) -> str:
    name = model_id.split("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def export_catalog(catalog_root: Path) -> int:
    source_root = Path(str(resources.files("aisimulate_core") / "model_configs"))
    destination_root = catalog_root / "models"
    destination_root.mkdir(parents=True, exist_ok=True)
    exported = 0
    used_names: dict[str, str] = {}

    for source in sorted(source_root.glob("*_config.json")):
        model_id = model_id_from_filename(source)
        name = safe_name(model_id)
        previous = used_names.get(name)
        if previous is not None and previous != model_id:
            raise RuntimeError(f"BLIS catalog name collision: {previous!r} and {model_id!r} -> {name!r}")
        used_names[name] = model_id
        destination = destination_root / name
        destination.mkdir(parents=True, exist_ok=True)
        json.loads(source.read_text())
        shutil.copy2(source, destination / "config.json")
        (destination / "model.yaml").write_text(
            f"name: {name}\nsource:\n  provider: aisimulate\n  repo: {model_id}\n  revision: packaged-aisimulate-wheel\n"
        )
        exported += 1

    if exported == 0:
        raise RuntimeError(f"no AISimulate model configs found in {source_root}")
    return exported


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <blis-catalog-root>")
    count = export_catalog(Path(sys.argv[1]))
    print(f"exported {count} AISimulate model configs into {sys.argv[1]}/models")
