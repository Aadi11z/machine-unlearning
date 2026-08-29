from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


CANONICAL_BASELINE_ID = "cifar100_canonical_v1"
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class BaselinePaths:
    root: Path
    development: Path
    pilots: Path
    final_fit: Path
    promoted: Path

    @property
    def checkpoints(self) -> Path:
        return self.final_fit / "checkpoints"

    @property
    def metrics(self) -> Path:
        return self.final_fit / "metrics"

    @property
    def manifests(self) -> Path:
        return self.final_fit / "manifests"


def resolve_baseline_paths(
    output_root: str | Path,
    *,
    baseline_id: str = CANONICAL_BASELINE_ID,
    run_id: str | None = None,
) -> BaselinePaths:
    if run_id is not None and not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must be 1-64 characters using letters, numbers, '.', '_' or '-'"
        )
    root = Path(output_root) / "cifar100" / "canonical"
    run_suffix = Path(run_id) if run_id is not None else None
    return BaselinePaths(
        root=root,
        development=root / "development" / run_suffix if run_suffix else root / "development",
        pilots=root / "pilots" / run_suffix if run_suffix else root / "pilots",
        final_fit=root / baseline_id,
        promoted=root / "promoted",
    )


def prepare_baseline_output_dirs(
    output_root: str | Path,
    *,
    baseline_id: str = CANONICAL_BASELINE_ID,
    run_id: str | None = None,
    allow_existing: bool = False,
) -> BaselinePaths:
    paths = resolve_baseline_paths(
        output_root, baseline_id=baseline_id, run_id=run_id
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
        paths.manifests,
        paths.promoted,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths
