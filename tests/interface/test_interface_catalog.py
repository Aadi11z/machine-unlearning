from __future__ import annotations

import json
from pathlib import Path

import pytest

from interface.catalog import ArtifactCatalog
from unml.manifest import (
    build_baseline_manifest,
    sha256_file,
    write_baseline_manifest,
)
from unml.prompts import resolve_prompt_contract


def _canonical_fixture(output_root: Path) -> tuple[Path, dict]:
    root = output_root / "cifar100" / "baseline"
    checkpoint = root / "checkpoints" / "finetuned_best.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"canonical")
    contract = resolve_prompt_contract("cifar100")
    manifest = build_baseline_manifest(
        baseline_id="baseline-v1",
        dataset="cifar100",
        split={"split_id": "canonical-split", "digest": "split-digest"},
        model_config={"model_name": "clip", "adapter_type": "vision_lora"},
        prompt_contract={"version": contract.version, "digest": contract.digest},
        checkpoints={"checkpoint": checkpoint},
        metrics={
            "class_names": [f"class-{index}" for index in range(100)],
            "final_metrics": {"test_all_acc": 0.8781},
        },
        artifact_root=root,
    )
    write_baseline_manifest(root / "manifest.json", manifest)
    return checkpoint, manifest


def test_catalog_resolves_only_verified_canonical_baseline(tmp_path: Path) -> None:
    legacy = (
        tmp_path
        / "outputs"
        / "cifar100"
        / "old_request"
        / "baseline_2000"
        / "checkpoints"
        / "finetuned_best.pt"
    )
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy")
    catalog = ArtifactCatalog(output_root=tmp_path / "outputs")
    assert catalog.baseline_checkpoint() is None

    canonical, _ = _canonical_fixture(tmp_path / "outputs")
    catalog = ArtifactCatalog(output_root=tmp_path / "outputs")
    assert catalog.baseline_checkpoint() == canonical
    assert catalog.baseline_identity() == {
        "baseline_id": "baseline-v1",
        "baseline_sha256": sha256_file(canonical),
    }
    assert len(catalog.baseline_class_names()) == 100


def test_legacy_baseline_override_uses_dataset_vocabulary(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.write_bytes(b"legacy")
    catalog = ArtifactCatalog(
        output_root=tmp_path / "outputs",
        baseline_checkpoint_path=checkpoint,
    )
    assert catalog.baseline_checkpoint() == checkpoint
    assert catalog.baseline_identity() == {
        "baseline_id": "legacy_override",
        "baseline_sha256": sha256_file(checkpoint),
    }
    assert len(catalog.baseline_class_names()) == 100
    assert catalog.baseline_class_names()[70] == "rose"


def test_catalog_discovers_legacy_precomputed_candidate(tmp_path: Path) -> None:
    _canonical_fixture(tmp_path / "outputs")
    checkpoint = (
        tmp_path
        / "outputs"
        / "cifar100"
        / "archive"
        / "legacy"
        / "rose_selective"
        / "unlearn_ga_kl_200"
        / "checkpoints"
        / "unlearn_ga_kl.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"legacy rose candidate")

    candidate = ArtifactCatalog(output_root=tmp_path / "outputs").precomputed_candidate(
        class_id=70, method="ga_kl", steps=200
    )

    assert candidate is not None
    assert candidate.checkpoint_path == checkpoint
    assert candidate.source == "precomputed"


def test_catalog_builds_reference_rows_from_verified_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, canonical = _canonical_fixture(tmp_path / "outputs")
    oracle_root = (
        tmp_path
        / "outputs"
        / "cifar100"
        / "oracle"
        / "run-1"
        / "flowers_superclass"
    )
    checkpoint = oracle_root / "checkpoints" / "retrained_best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"oracle")
    manifest = {
        "schema": "unml-retraining-oracle-manifest-v1",
        "oracle_id": "cifar100_retraining_oracle_v1",
        "dataset": "cifar100",
        "request": {
            "request_name": "flowers_superclass",
            "request_type": "superclass",
            "forget_class_names": ["orchid", "poppy", "rose", "sunflower", "tulip"],
            "test_forget_count": 500,
            "test_retain_count": 9500,
        },
        "canonical_contract": {
            "baseline_id": canonical["baseline_id"],
            "manifest_sha256": sha256_file(
                tmp_path
                / "outputs"
                / "cifar100"
                / "baseline"
                / "manifest.json"
            ),
            "prompt_digest": canonical["prompt_contract"]["digest"],
            "split_digest": canonical["split"]["digest"],
        },
        "prompt_contract": canonical["prompt_contract"],
        "model_config": canonical["model_config"],
        "artifacts": {"checkpoint": {"path": "checkpoints/retrained_best.pt"}},
        "metrics": {
            "final_metrics": {"test_all_acc": 0.8493, "test_retain_acc": 0.8732631578947369}
        },
        "comparison": {
            "baseline_id": canonical["baseline_id"],
            "checkpoints": {
                "canonical_sha256": canonical["artifacts"]["checkpoint"]["sha256"]
            },
            "evaluations": {
                "canonical": {
                    "test_all": {"accuracy": 0.8781},
                    "test_retain": {"accuracy": 0.9},
                    "test_forget": {"accuracy": 0.462},
                },
                "oracle": {
                    "test_all": {"accuracy": 0.8493},
                    "test_retain": {"accuracy": 0.8732631578947369},
                    "test_forget": {"accuracy": 0.394},
                },
            }
        },
    }
    (oracle_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    index_path = oracle_root.parents[1] / "promoted.json"
    index_path.write_text(
        json.dumps(
            {
                "schema": "unml-retraining-oracle-index-v1",
                "references": {
                    "flowers_superclass": "run-1/flowers_superclass/manifest.json"
                },
            }
        ),
        encoding="utf-8",
    )
    historical = (
        oracle_root.parents[1]
        / "run-0"
        / "flowers_superclass"
        / "manifest.json"
    )
    historical.parent.mkdir(parents=True)
    historical.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "interface.catalog.verify_retraining_oracle_manifest",
        lambda *_args, **_kwargs: None,
    )

    reference = ArtifactCatalog(output_root=tmp_path / "outputs").reference_artifacts()[0]
    assert reference.request_name == "flowers_superclass"
    assert reference.metric_rows[0]["delta"] == "-2.88 pp"
    assert reference.metric_rows[2]["oracle"] == "39.40%"
    assert reference.metric_rows[2]["canonical"] == "46.20%"
