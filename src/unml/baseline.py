from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CANONICAL_BASELINE_ID = "cifar100_canonical_v1"


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
) -> BaselinePaths:
    root = Path(output_root) / "cifar100" / "canonical"
    return BaselinePaths(
        root=root,
        development=root / "development",
        pilots=root / "pilots",
        final_fit=root / baseline_id,
        promoted=root / "promoted",
    )


def prepare_baseline_output_dirs(
    output_root: str | Path,
    *,
    baseline_id: str = CANONICAL_BASELINE_ID,
    allow_existing: bool = False,
) -> BaselinePaths:
    paths = resolve_baseline_paths(output_root, baseline_id=baseline_id)
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
