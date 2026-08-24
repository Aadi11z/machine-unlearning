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
) -> dict[str, Any]:
    if not baseline_id.strip():
        raise ValueError("baseline_id must not be empty")
    if not checkpoints:
        raise ValueError("At least one baseline checkpoint is required")
    artifacts = {
        role: artifact_record(path) for role, path in sorted(checkpoints.items())
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
    expected_prompt = manifest["prompt_contract"]
    expected_model = manifest["model_config"]
    for role, record in manifest["artifacts"].items():
        path = root_path / str(record["path"])
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
