from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from unml.study import (
    aggregate_study,
    bootstrap_mean_interval,
    load_study_records,
    paired_study_differences,
    write_study_artifacts,
)
from scripts.run_study import cell_paths, write_manifest


METRICS = ("target_test_acc", "utility_test_all")
METHODS = ("h_tgsd", "ga_kl")


def _comparison(request: str, seed: int, path: Path) -> None:
    rows = []
    for method, offset in (("h_tgsd", 0.0), ("ga_kl", 0.1)):
        rows.append(
            {
                "model": method,
                "dataset": "cifar100",
                "request": request,
                "seed": seed,
                "git_commit": "abc123",
                "gpu_name": "NVIDIA A100-SXM4-80GB",
                "cuda_version": "12.4",
                "runtime_seconds": 10.0 + seed,
                "peak_gpu_memory_mb": 2000.0,
                "trainable_parameter_count": 4096,
                "checkpoint_size_bytes": 8192,
                "target_test_acc": seed / 1000.0 + offset,
                "utility_test_all": 0.9 - offset,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _manifest(tmp_path: Path) -> Path:
    rows = []
    for request in ("flowers_superclass", "rose_selective"):
        for seed in (42, 123, 456):
            path = tmp_path / f"{request}_{seed}.csv"
            _comparison(request, seed, path)
            rows.append(
                {
                    "request": request,
                    "seed": seed,
                    "comparison_path": path.name,
                }
            )
    manifest = tmp_path / "study_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def test_bootstrap_interval_is_deterministic_and_ordered() -> None:
    first = bootstrap_mean_interval(
        [1.0, 2.0, 3.0], samples=500, confidence=0.95, seed=7
    )
    second = bootstrap_mean_interval(
        [1.0, 2.0, 3.0], samples=500, confidence=0.95, seed=7
    )
    assert first == second
    assert first[0] <= 2.0 <= first[1]


def test_study_load_aggregate_pair_and_write(tmp_path: Path) -> None:
    records = load_study_records(
        _manifest(tmp_path),
        expected_dataset="cifar100",
        expected_requests=("flowers_superclass", "rose_selective"),
        expected_seeds=(42, 123, 456),
        required_methods=METHODS,
        required_metrics=METRICS,
    )
    aggregate = aggregate_study(
        records,
        metrics=METRICS,
        bootstrap_samples=500,
        confidence=0.95,
        bootstrap_seed=1,
    )
    paired = paired_study_differences(
        records,
        primary_method="h_tgsd",
        baseline_methods=("ga_kl",),
        metrics=METRICS,
        bootstrap_samples=500,
        confidence=0.95,
        bootstrap_seed=1,
    )
    paths = write_study_artifacts(
        records,
        aggregate,
        paired,
        tmp_path / "summary",
        {"status": "passed"},
    )

    assert len(records) == 12
    assert set(aggregate["n"]) == {3}
    target_delta = paired[
        paired["metric"] == "target_test_acc"
    ]["mean_delta"]
    assert target_delta.tolist() == pytest.approx([-0.1, -0.1])
    assert Path(paths["aggregate_markdown"]).exists()
    assert "\\begin{longtable}" in Path(
        paths["aggregate_latex"]
    ).read_text()
    assert "\\begin{longtable}" in Path(paths["paired_latex"]).read_text()
    assert json.loads(Path(paths["validation_json"]).read_text()) == {
        "status": "passed"
    }


def test_study_rejects_missing_seed_and_mixed_gpu(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    rows = pd.read_csv(manifest).iloc[:-1]
    rows.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="matrix mismatch"):
        load_study_records(
            manifest,
            expected_dataset="cifar100",
            expected_requests=("flowers_superclass", "rose_selective"),
            expected_seeds=(42, 123, 456),
            required_methods=METHODS,
            required_metrics=METRICS,
        )


def test_manifest_writer_requires_every_isolated_cell(tmp_path: Path) -> None:
    root = tmp_path / "cifar100"
    requests = ["flowers_superclass"]
    seeds = [42, 123]
    first = cell_paths(root, requests[0], seeds[0])["comparison"]
    first.parent.mkdir(parents=True)
    first.write_text("model\nh_tgsd\n")
    destination = root / "study_manifest.csv"

    with pytest.raises(FileNotFoundError, match="missing comparisons"):
        write_manifest(root, requests, seeds, destination)

    second = cell_paths(root, requests[0], seeds[1])["comparison"]
    second.parent.mkdir(parents=True)
    second.write_text("model\nh_tgsd\n")
    write_manifest(root, requests, seeds, destination)

    manifest = pd.read_csv(destination)
    assert manifest["seed"].tolist() == seeds

    manifest = _manifest(tmp_path)
    comparison = pd.read_csv(tmp_path / "flowers_superclass_42.csv")
    comparison.loc[0, "gpu_name"] = "NVIDIA H100 80GB"
    comparison.to_csv(tmp_path / "flowers_superclass_42.csv", index=False)
    with pytest.raises(ValueError, match="mixes GPU"):
        load_study_records(
            manifest,
            expected_dataset="cifar100",
            expected_requests=("flowers_superclass", "rose_selective"),
            expected_seeds=(42, 123, 456),
            required_methods=METHODS,
            required_metrics=METRICS,
        )
