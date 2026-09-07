from __future__ import annotations

import copy
import json

import pytest

from unml.manifest import (
    artifact_record,
    build_baseline_manifest,
    build_retraining_oracle_manifest,
    sha256_file,
    validate_baseline_identity,
    verify_manifest_artifacts,
    verify_retraining_oracle_manifest,
    write_baseline_manifest,
    write_retraining_oracle_manifest,
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


def test_retraining_oracle_manifest_binds_all_artifacts(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoints" / "retrained_best.pt"
    metrics_path = tmp_path / "metrics" / "retrain_metrics.json"
    split_path = tmp_path / "splits" / "flowers_superclass.json"
    comparison_path = tmp_path / "metrics" / "reference_comparison.json"
    for path, content in (
        (checkpoint, b"checkpoint"),
        (metrics_path, b'{"final_metrics": {"test_all_acc": 0.8}}'),
        (
            split_path,
            json.dumps(
                {
                    "request_name": "flowers_superclass",
                    "request_type": "superclass",
                    "forget_classes": [0],
                    "class_names": ["orchid"],
                    "test_forget_indices": [0],
                    "test_retain_indices": [1],
                }
            ).encode(),
        ),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    prompt = {"version": "openai_cifar100_v1", "digest": "prompt-hash"}
    model = {"model_name": "clip", "adapter_type": "vision_lora"}
    comparison = {
        "schema": "unml-reference-comparison-v1",
        "dataset": "cifar100",
        "baseline_id": "cifar100_canonical_v1",
        "oracle_id": "cifar100_retraining_oracle_v1",
        "request_name": "flowers_superclass",
        "request_type": "superclass",
        "forget_classes": [0],
        "prompt_contract": prompt,
        "split_sha256": sha256_file(split_path),
        "checkpoints": {
            "canonical_sha256": "canonical-hash",
            "oracle_sha256": sha256_file(checkpoint),
        },
        "evaluations": {
            model_name: {
                partition: {"accuracy": 0.5}
                for partition in ("test_all", "test_retain", "test_forget")
            }
            for model_name in ("canonical", "oracle")
        },
    }
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    monkeypatch.setattr(
        "unml.manifest.read_checkpoint_payload",
        lambda _path: {
            "model_config": model,
            "extra": {"prompt_contract": prompt},
        },
    )
    manifest = build_retraining_oracle_manifest(
        oracle_id="cifar100_retraining_oracle_v1",
        dataset="cifar100",
        request={
            "request_name": "flowers_superclass",
            "request_type": "superclass",
            "forget_classes": [0],
            "forget_class_names": ["orchid"],
            "test_forget_count": 1,
            "test_retain_count": 1,
        },
        canonical_contract={"baseline_id": "cifar100_canonical_v1"},
        model_config=model,
        prompt_contract=prompt,
        checkpoint=checkpoint,
        metrics_path=metrics_path,
        split_path=split_path,
        comparison_path=comparison_path,
        metrics={"final_metrics": {"test_all_acc": 0.8}},
        comparison=comparison,
        artifact_root=tmp_path,
    )
    destination = write_retraining_oracle_manifest(
        tmp_path / "manifest.json", manifest
    )
    verify_retraining_oracle_manifest(manifest, root=tmp_path)
    assert destination.is_file()

    incomplete = copy.deepcopy(manifest)
    incomplete["comparison"]["evaluations"] = {}
    comparison_path.write_text(
        json.dumps(incomplete["comparison"]), encoding="utf-8"
    )
    incomplete["artifacts"]["comparison"] = artifact_record(
        comparison_path, root=tmp_path
    )
    with pytest.raises(ValueError, match="lacks canonical evaluation"):
        verify_retraining_oracle_manifest(incomplete, root=tmp_path)
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    mislabeled = copy.deepcopy(manifest)
    mislabeled["request"]["request_name"] = "wrong_request"
    with pytest.raises(ValueError, match="request does not match"):
        verify_retraining_oracle_manifest(mislabeled, root=tmp_path)

    split_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        verify_retraining_oracle_manifest(manifest, root=tmp_path)
