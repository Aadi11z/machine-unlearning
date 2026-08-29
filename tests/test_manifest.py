from __future__ import annotations

import copy

import pytest

from unml.manifest import (
    build_baseline_manifest,
    validate_baseline_identity,
    verify_manifest_artifacts,
    write_baseline_manifest,
)


def test_baseline_manifest_round_trip_and_hash_verification(tmp_path) -> None:
    checkpoint = tmp_path / "finetuned_best.bin"
    checkpoint.write_bytes(b"checkpoint")
    manifest = build_baseline_manifest(
        baseline_id="cifar100_canonical_v1",
        dataset="cifar100",
        split={"split_id": "split-v1", "digest": "split-hash"},
        model_config={"model_name": "clip", "adapter_type": "vision_lora"},
        prompt_contract={"version": "openai_cifar100_v1", "digest": "prompt-hash"},
        checkpoints={"best": checkpoint},
        metrics={"retain_val_acc": 0.8},
    )
    path = write_baseline_manifest(tmp_path / "manifest.json", manifest)
    verify_manifest_artifacts(manifest, root=tmp_path)
    assert path.is_file()

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        verify_manifest_artifacts(manifest, root=tmp_path)


def test_manifest_rejects_missing_required_fields(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"x")
    manifest = build_baseline_manifest(
        baseline_id="baseline",
        dataset="cifar100",
        split={},
        model_config={},
        prompt_contract={},
        checkpoints={"best": checkpoint},
        metrics={},
    )
    broken = copy.deepcopy(manifest)
    del broken["prompt_contract"]
    with pytest.raises(ValueError, match="missing fields"):
        write_baseline_manifest(tmp_path / "broken.json", broken)


def test_manifest_records_checkpoint_path_relative_to_artifact_root(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "finetuned_best.bin"
    checkpoint.write_bytes(b"checkpoint")
    manifest = build_baseline_manifest(
        baseline_id="canonical",
        dataset="cifar100",
        split={},
        model_config={},
        prompt_contract={},
        checkpoints={"checkpoint": checkpoint},
        metrics={},
        artifact_root=tmp_path,
    )
    assert manifest["artifacts"]["checkpoint"]["path"] == "checkpoints/finetuned_best.bin"
    verify_manifest_artifacts(manifest, root=tmp_path)

def test_manifest_identity_requires_id_and_checkpoint_digest(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    manifest = build_baseline_manifest(
        baseline_id="canonical",
        dataset="cifar100",
        split={},
        model_config={},
        prompt_contract={},
        checkpoints={"best": checkpoint},
        metrics={},
    )
    validate_baseline_identity(
        manifest,
        baseline_id="canonical",
        checkpoint_path=checkpoint,
    )
    with pytest.raises(ValueError, match="id mismatch"):
        validate_baseline_identity(
            manifest, baseline_id="wrong", checkpoint_path=checkpoint
        )
