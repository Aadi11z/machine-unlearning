from __future__ import annotations

import pytest

from unml.baseline import (
    CANONICAL_BASELINE_ID,
    prepare_baseline_output_dirs,
    resolve_baseline_paths,
)


def test_prepare_baseline_output_dirs_creates_contract(tmp_path) -> None:
    paths = prepare_baseline_output_dirs(tmp_path)
    assert paths.final_fit.name == CANONICAL_BASELINE_ID
    assert paths.checkpoints.is_dir()
    assert paths.metrics.is_dir()
    assert paths.manifests.is_dir()
    assert paths.promoted.is_dir()


def test_prepare_baseline_output_dirs_rejects_collision(tmp_path) -> None:
    paths = resolve_baseline_paths(tmp_path)
    paths.final_fit.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_baseline_output_dirs(tmp_path)
