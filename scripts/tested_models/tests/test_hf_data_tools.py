from __future__ import annotations

import json
from pathlib import Path

import download_ground_truth
import publish_ground_truth
import pytest


def test_download_rejects_non_immutable_revision(monkeypatch) -> None:
    monkeypatch.setattr(download_ground_truth, "snapshot_download", lambda **_: pytest.fail("must not download"))
    monkeypatch.setattr(download_ground_truth, "parse_args", lambda: type("Args", (), {
        "repo_id": "org/data", "revision": "main", "output_dir": Path("data")
    })())
    with pytest.raises(SystemExit, match="full 40-character"):
        download_ground_truth.main()


def test_download_rejects_unsafe_pair_id(tmp_path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "manifest.json").write_text(json.dumps({"pairs": [{"id": "../escape", "path": "pairs/x/ground_truth.parquet"}]}))
    monkeypatch.setattr(download_ground_truth, "snapshot_download", lambda **_: str(snapshot))
    monkeypatch.setattr(download_ground_truth, "parse_args", lambda: type("Args", (), {
        "repo_id": "org/data", "revision": "a" * 40, "output_dir": tmp_path / "output"
    })())
    with pytest.raises(ValueError, match="filesystem pair id"):
        download_ground_truth.main()


def test_publish_rejects_output_input_overlap(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    manifest.write_text(json.dumps({"pairs": []}))
    with pytest.raises(ValueError, match="input path"):
        publish_ground_truth.prepare(manifest, datasets, tmp_path)


def test_publish_rejects_empty_manifest_before_cleanup(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    datasets = tmp_path / "datasets"
    output = tmp_path / "output"
    datasets.mkdir()
    output.mkdir()
    marker = output / "must-survive"
    marker.write_text("keep")
    manifest.write_text(json.dumps({"pairs": []}))

    with pytest.raises(ValueError, match="at least one pair"):
        publish_ground_truth.prepare(manifest, datasets, output)
    assert marker.read_text() == "keep"
