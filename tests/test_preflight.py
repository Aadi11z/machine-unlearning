from __future__ import annotations

import json
from pathlib import Path

import pytest

from unml.preflight import (
    cache_manifest_path,
    validate_cache_manifest,
    validate_scratch_layout,
    validate_writable_directory,
    write_cache_manifest,
)


def test_cache_manifest_round_trip(tmp_path: Path) -> None:
    path = write_cache_manifest(
        model_name="openai/clip-vit-base-patch16",
        dataset_name="cifar100",
        hf_home=tmp_path,
    )

    resolved_path, payload = validate_cache_manifest(
        model_name="openai/clip-vit-base-patch16",
        dataset_name="cifar100",
        hf_home=tmp_path,
    )

    assert resolved_path == path
    assert payload["offline_reload_verified"] is True
    assert path == cache_manifest_path(
        "openai/clip-vit-base-patch16", tmp_path
    )


def test_cache_manifest_rejects_dataset_mismatch(tmp_path: Path) -> None:
    path = write_cache_manifest(
        model_name="openai/clip-vit-base-patch16",
        dataset_name="cifar100",
        hf_home=tmp_path,
    )
    payload = json.loads(path.read_text())
    payload["dataset"] = "cifar10"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="manifest mismatch"):
        validate_cache_manifest(
            model_name="openai/clip-vit-base-patch16",
            dataset_name="cifar100",
            hf_home=tmp_path,
        )


def test_scratch_layout_rejects_heavy_path_outside_root(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    outside = tmp_path / "home" / "outputs"

    with pytest.raises(ValueError, match="must be under scratch"):
        validate_scratch_layout(
            {"data": scratch / "data", "outputs": outside}, scratch
        )


def test_writable_directory_check_creates_no_persistent_probe(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "outputs"

    resolved = validate_writable_directory(destination)

    assert resolved == destination.resolve()
    assert list(destination.iterdir()) == []
