from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
from typing import Any, Mapping

from .manifest import sha256_file, verify_manifest_artifacts


BASELINE_MANIFEST_ENV = "UNML_BASELINE_MANIFEST"
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class BaselinePaths:
    root: Path
    development: Path
    pilots: Path
    final_fit: Path

    @property
    def checkpoints(self) -> Path:
        return self.final_fit / "checkpoints"

    @property
    def metrics(self) -> Path:
        return self.final_fit / "metrics"

@dataclass(frozen=True)
class BaselineReference:
    dataset: str
    baseline_id: str
    package_root: Path
    manifest_path: Path
    manifest_sha256: str
    final_checkpoint: Path
    final_checkpoint_sha256: str
    manifest: Mapping[str, Any]


def _safe_relative(root: Path, value: str, *, field: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or not value or ".." in candidate.parts:
        raise ValueError(f"{field} must be a relative path within the canonical root")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{field} escapes the canonical root")
    return resolved


def resolve_baseline(
    manifest_path: str | Path, *, dataset: str = "cifar100"
) -> BaselineReference:
    """Load and verify the configured baseline package once at process startup."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Baseline manifest missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid baseline manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("dataset") != dataset:
        raise ValueError("Baseline manifest dataset does not match configuration")
    verify_manifest_artifacts(manifest, root=manifest_path.parent)
    artifacts = manifest.get("artifacts", {})
    checkpoint = artifacts.get("final_checkpoint", artifacts.get("checkpoint"))
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Baseline manifest lacks a checkpoint artifact")
    final_checkpoint = _safe_relative(
        manifest_path.parent, str(checkpoint.get("path", "")), field="checkpoint path"
    )
    return BaselineReference(
        dataset=dataset,
        baseline_id=str(manifest["baseline_id"]),
        package_root=manifest_path.parent,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        final_checkpoint=final_checkpoint,
        final_checkpoint_sha256=sha256_file(final_checkpoint),
        manifest=manifest,
    )


def configured_baseline_manifest(
    output_root: str | Path, *, dataset: str = "cifar100"
) -> Path:
    configured = os.environ.get(BASELINE_MANIFEST_ENV)
    return Path(configured) if configured else Path(output_root) / dataset / "baseline" / "manifest.json"


def resolve_baseline_paths(
    output_root: str | Path,
    *,
    run_id: str | None = None,
) -> BaselinePaths:
    if run_id is not None and not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must be 1-64 characters using letters, numbers, '.', '_' or '-'"
        )
    root = Path(output_root) / "cifar100"
    run_suffix = Path(run_id) if run_id is not None else None
    return BaselinePaths(
        root=root,
        development=root / "development" / run_suffix if run_suffix else root / "development",
        pilots=root / "pilots" / run_suffix if run_suffix else root / "pilots",
        final_fit=root / "baseline",
    )


def prepare_baseline_output_dirs(
    output_root: str | Path,
    *,
    run_id: str | None = None,
    allow_existing: bool = False,
) -> BaselinePaths:
    paths = resolve_baseline_paths(
        output_root, run_id=run_id
    )
    if paths.final_fit.exists() and not allow_existing:
        raise FileExistsError(
            f"Final-fit baseline directory already exists: {paths.final_fit}"
        )
    for directory in (
        paths.development,
        paths.pilots,
        paths.checkpoints,
        paths.metrics,
        paths.final_fit,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths
