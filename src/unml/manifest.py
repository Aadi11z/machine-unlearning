"""Immutable baseline provenance manifests and artifact verification."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .model import read_checkpoint_payload


BASELINE_MANIFEST_SCHEMA = "unml-baseline-manifest-v1"
RETRAINING_ORACLE_MANIFEST_SCHEMA = "unml-retraining-oracle-manifest-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    artifact = Path(path)
    relative = artifact.name if root is None else str(artifact.relative_to(root))
    return {
        "path": relative,
        "size_bytes": artifact.stat().st_size,
        "sha256": sha256_file(artifact),
    }


def build_baseline_manifest(
    *,
    baseline_id: str,
    dataset: str,
    split: Mapping[str, Any],
    model_config: Mapping[str, Any],
    prompt_contract: Mapping[str, Any],
    checkpoints: Mapping[str, str | Path],
    metrics: Mapping[str, Any],
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    if not baseline_id.strip():
        raise ValueError("baseline_id must not be empty")
    if not checkpoints:
        raise ValueError("At least one baseline checkpoint is required")
    artifacts = {
        role: artifact_record(path, root=artifact_root)
        for role, path in sorted(checkpoints.items())
    }
    return {
        "schema": BASELINE_MANIFEST_SCHEMA,
        "baseline_id": baseline_id,
        "dataset": dataset,
        "split": {
            "split_id": split.get("split_id"),
            "digest": split.get("digest"),
        },
        "model_config": dict(model_config),
        "prompt_contract": dict(prompt_contract),
        "artifacts": artifacts,
        "metrics": dict(metrics),
    }


def validate_baseline_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema", "baseline_id", "dataset", "split", "model_config",
        "prompt_contract", "artifacts", "metrics",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Baseline manifest missing fields: {missing}")
    if manifest["schema"] != BASELINE_MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported baseline manifest schema: {manifest['schema']!r}")
    if not isinstance(manifest["artifacts"], Mapping) or not manifest["artifacts"]:
        raise ValueError("Baseline manifest must contain artifacts")
    for role, record in manifest["artifacts"].items():
        if not isinstance(record, Mapping) or not record.get("sha256"):
            raise ValueError(f"Artifact {role!r} lacks a sha256 digest")


def verify_manifest_artifacts(
    manifest: Mapping[str, Any], *, root: str | Path
) -> None:
    validate_baseline_manifest(manifest)
    root_path = Path(root)
    resolved_root = root_path.resolve()
    expected_prompt = manifest["prompt_contract"]
    expected_model = manifest["model_config"]
    for role, record in manifest["artifacts"].items():
        relative = Path(str(record["path"]))
        if relative.is_absolute() or not str(relative) or ".." in relative.parts:
            raise ValueError(f"Manifest artifact {role!r} path must stay within its root")
        path = (root_path / relative).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError(f"Manifest artifact {role!r} path escapes its root")
        if not path.is_file():
            raise FileNotFoundError(f"Manifest artifact {role!r} is missing: {path}")
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size != int(record["size_bytes"]) or actual_hash != record["sha256"]:
            raise ValueError(f"Manifest artifact {role!r} does not match: {path}")
        if path.suffix == ".pt":
            payload = read_checkpoint_payload(path)
            stored_extra = payload.get("extra", {})
            stored_prompt = stored_extra.get("prompt_contract", {})
            if expected_prompt.get("digest") and stored_prompt.get("digest") != expected_prompt["digest"]:
                raise ValueError(f"Checkpoint {role!r} prompt contract does not match manifest")
            stored_config = payload.get("model_config", {})
            for key in ("model_name", "adapter_type", "lora_rank", "lora_layers", "lora_targets"):
                if key in expected_model and stored_config.get(key) != expected_model[key]:
                    raise ValueError(f"Checkpoint {role!r} model config does not match manifest")


def write_baseline_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    validate_baseline_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return destination
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_retraining_oracle_manifest(
    *,
    oracle_id: str,
    dataset: str,
    request: Mapping[str, Any],
    canonical_contract: Mapping[str, Any],
    model_config: Mapping[str, Any],
    prompt_contract: Mapping[str, Any],
    checkpoint: str | Path,
    metrics_path: str | Path,
    split_path: str | Path,
    comparison_path: str | Path,
    metrics: Mapping[str, Any],
    comparison: Mapping[str, Any],
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Bind one target-specific retraining oracle to its immutable inputs."""
    if not oracle_id.strip():
        raise ValueError("oracle_id must not be empty")
    return {
        "schema": RETRAINING_ORACLE_MANIFEST_SCHEMA,
        "oracle_id": oracle_id,
        "dataset": dataset,
        "request": {
            "request_name": request.get("request_name"),
            "request_type": request.get("request_type"),
            "forget_classes": list(request.get("forget_classes", [])),
            "forget_class_names": list(request.get("forget_class_names", [])),
            "test_forget_count": request.get("test_forget_count"),
            "test_retain_count": request.get("test_retain_count"),
        },
        "canonical_contract": dict(canonical_contract),
        "model_config": dict(model_config),
        "prompt_contract": dict(prompt_contract),
        "artifacts": {
            "checkpoint": artifact_record(checkpoint, root=artifact_root),
            "metrics": artifact_record(metrics_path, root=artifact_root),
            "split": artifact_record(split_path, root=artifact_root),
            "comparison": artifact_record(comparison_path, root=artifact_root),
        },
        "metrics": dict(metrics),
        "comparison": dict(comparison),
    }


def validate_retraining_oracle_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema", "oracle_id", "dataset", "request", "canonical_contract",
        "model_config", "prompt_contract", "artifacts", "metrics", "comparison",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Retraining oracle manifest missing fields: {missing}")
    if manifest["schema"] != RETRAINING_ORACLE_MANIFEST_SCHEMA:
        raise ValueError(
            f"Unsupported retraining oracle manifest schema: {manifest['schema']!r}"
        )
    request = manifest["request"]
    if not isinstance(request, Mapping) or not request.get("request_name"):
        raise ValueError("Retraining oracle manifest lacks a request identity")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise ValueError("Retraining oracle manifest must contain artifacts")
    for role in ("checkpoint", "metrics", "split", "comparison"):
        record = artifacts.get(role)
        if not isinstance(record, Mapping) or not record.get("sha256"):
            raise ValueError(f"Retraining oracle artifact {role!r} lacks a digest")


def verify_retraining_oracle_manifest(
    manifest: Mapping[str, Any], *, root: str | Path
) -> None:
    validate_retraining_oracle_manifest(manifest)
    root_path = Path(root)
    for role, record in manifest["artifacts"].items():
        path = root_path / str(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Oracle artifact {role!r} is missing: {path}")
        if (
            path.stat().st_size != int(record["size_bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"Oracle artifact {role!r} does not match: {path}")

    metrics_path = root_path / str(manifest["artifacts"]["metrics"]["path"])
    split_path = root_path / str(manifest["artifacts"]["split"]["path"])
    comparison_path = root_path / str(
        manifest["artifacts"]["comparison"]["path"]
    )
    raw_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    raw_split = json.loads(split_path.read_text(encoding="utf-8"))
    raw_comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if raw_metrics != manifest["metrics"]:
        raise ValueError("Oracle manifest metrics do not match the metrics artifact")
    if raw_comparison != manifest["comparison"]:
        raise ValueError(
            "Oracle manifest comparison does not match the comparison artifact"
        )
    request = manifest["request"]
    forget_classes = [int(value) for value in raw_split.get("forget_classes", [])]
    class_names = [str(value) for value in raw_split.get("class_names", [])]
    expected_request = {
        "request_name": raw_split.get("request_name"),
        "request_type": raw_split.get("request_type"),
        "forget_classes": forget_classes,
        "forget_class_names": [class_names[index] for index in forget_classes],
        "test_forget_count": len(raw_split.get("test_forget_indices", [])),
        "test_retain_count": len(raw_split.get("test_retain_indices", [])),
    }
    if request != expected_request:
        raise ValueError("Oracle manifest request does not match the split artifact")

    comparison = manifest["comparison"]
    if (
        comparison.get("schema") != "unml-reference-comparison-v1"
        or comparison.get("dataset") != manifest["dataset"]
        or comparison.get("oracle_id") != manifest["oracle_id"]
        or comparison.get("request_name") != request["request_name"]
        or comparison.get("request_type") != request["request_type"]
        or comparison.get("forget_classes") != request["forget_classes"]
        or comparison.get("split_sha256")
        != manifest["artifacts"]["split"]["sha256"]
        or comparison.get("checkpoints", {}).get("oracle_sha256")
        != manifest["artifacts"]["checkpoint"]["sha256"]
        or comparison.get("prompt_contract", {}).get("digest")
        != manifest["prompt_contract"].get("digest")
    ):
        raise ValueError("Oracle comparison identity does not match manifest")
    evaluations = comparison.get("evaluations")
    if not isinstance(evaluations, Mapping):
        raise ValueError("Oracle comparison lacks evaluations")
    for model_name in ("canonical", "oracle"):
        model_evaluation = evaluations.get(model_name)
        if not isinstance(model_evaluation, Mapping):
            raise ValueError(f"Oracle comparison lacks {model_name} evaluation")
        for partition in ("test_all", "test_retain", "test_forget"):
            result = model_evaluation.get(partition)
            accuracy = result.get("accuracy") if isinstance(result, Mapping) else None
            if (
                isinstance(accuracy, bool)
                or not isinstance(accuracy, (int, float))
                or not 0.0 <= float(accuracy) <= 1.0
            ):
                raise ValueError(
                    f"Oracle comparison lacks numeric {model_name}.{partition}.accuracy"
                )

    checkpoint_record = manifest["artifacts"]["checkpoint"]
    checkpoint = root_path / str(checkpoint_record["path"])
    payload = read_checkpoint_payload(checkpoint)
    stored_prompt = payload.get("extra", {}).get("prompt_contract", {})
    expected_prompt = manifest["prompt_contract"]
    if stored_prompt.get("digest") != expected_prompt.get("digest"):
        raise ValueError("Oracle checkpoint prompt contract does not match manifest")
    stored_config = payload.get("model_config", {})
    for key in ("model_name", "adapter_type", "lora_rank", "lora_layers", "lora_targets"):
        if key in manifest["model_config"] and stored_config.get(key) != manifest["model_config"][key]:
            raise ValueError("Oracle checkpoint model config does not match manifest")


def verify_retraining_oracle_canonical_contract(
    oracle: Mapping[str, Any],
    canonical: Mapping[str, Any],
    *,
    canonical_manifest_path: str | Path,
) -> None:
    """Require an oracle to match the exact canonical release and schedule."""
    canonical_contract = oracle.get("canonical_contract", {})
    comparison = oracle.get("comparison", {})
    canonical_checkpoint = canonical["artifacts"].get("checkpoint")
    mismatches: dict[str, Any] = {}
    expected_pairs = {
        "baseline_id": (canonical.get("baseline_id"), canonical_contract.get("baseline_id")),
        "manifest_sha256": (
            sha256_file(canonical_manifest_path),
            canonical_contract.get("manifest_sha256"),
        ),
        "prompt_digest": (
            canonical.get("prompt_contract", {}).get("digest"),
            canonical_contract.get("prompt_digest"),
        ),
        "oracle_prompt_digest": (
            canonical.get("prompt_contract", {}).get("digest"),
            oracle.get("prompt_contract", {}).get("digest"),
        ),
        "split_digest": (
            canonical.get("split", {}).get("digest"),
            canonical_contract.get("split_digest"),
        ),
        "comparison_baseline_id": (
            canonical.get("baseline_id"),
            comparison.get("baseline_id"),
        ),
        "comparison_checkpoint_sha256": (
            canonical_checkpoint.get("sha256")
            if isinstance(canonical_checkpoint, Mapping)
            else None,
            comparison.get("checkpoints", {}).get("canonical_sha256"),
        ),
        "model_config": (canonical.get("model_config"), oracle.get("model_config")),
    }
    for name, (expected, actual) in expected_pairs.items():
        if expected != actual:
            mismatches[name] = (expected, actual)

    canonical_metrics = canonical.get("metrics", {})
    oracle_metrics = oracle.get("metrics", {})
    canonical_config = canonical_metrics.get("config", {})
    oracle_config = oracle_metrics.get("config", {})
    fields = (
        "dataset_name", "model_name", "prompt_template", "adapter_rank",
        "adapter_alpha", "adapter_type", "lora_rank", "lora_alpha",
        "lora_layers", "lora_targets", "lora_dropout", "train_logit_scale",
        "precision", "gradient_checkpointing", "batch_size",
        "gradient_accumulation_steps", "lr", "weight_decay", "seed",
    )
    for field in fields:
        if (
            field in canonical_config
            and canonical_config.get(field) != oracle_config.get(field)
        ):
            mismatches[f"config.{field}"] = (
                canonical_config.get(field),
                oracle_config.get(field),
            )
    expected_initial = canonical_metrics.get("base_checkpoint_sha256")
    actual_initial = oracle_metrics.get("source_initial_checkpoint_sha256")
    if expected_initial and expected_initial != actual_initial:
        mismatches["initial_checkpoint_sha256"] = (expected_initial, actual_initial)
    expected_steps = canonical_metrics.get("global_steps")
    actual_steps = oracle_config.get("target_optimizer_steps")
    if expected_steps and expected_steps != actual_steps:
        mismatches["optimizer_steps"] = (expected_steps, actual_steps)
    if mismatches:
        raise ValueError(f"Oracle canonical contract mismatch: {mismatches}")


def write_retraining_oracle_manifest(
    path: str | Path, manifest: Mapping[str, Any]
) -> Path:
    validate_retraining_oracle_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return destination
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def validate_baseline_identity(
    manifest: Mapping[str, Any],
    *,
    baseline_id: str,
    checkpoint_path: str | Path,
) -> None:
    """Require consumers to use the exact configured id and checkpoint digest."""
    validate_baseline_manifest(manifest)
    if manifest["baseline_id"] != baseline_id:
        raise ValueError(
            f"Baseline id mismatch: expected {manifest['baseline_id']!r}, "
            f"received {baseline_id!r}"
        )
    checkpoint_record = manifest["artifacts"].get("checkpoint") or manifest[
        "artifacts"
    ].get("best")
    if not isinstance(checkpoint_record, Mapping):
        raise ValueError("Baseline manifest lacks a checkpoint artifact")
    actual_digest = sha256_file(checkpoint_path)
    if actual_digest != checkpoint_record.get("sha256"):
        raise ValueError(
            f"Baseline checkpoint digest mismatch for {checkpoint_path}"
        )
