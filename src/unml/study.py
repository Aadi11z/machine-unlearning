from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROVENANCE_COLUMNS = (
    "dataset",
    "request",
    "seed",
    "git_commit",
    "gpu_name",
    "cuda_version",
    "runtime_seconds",
    "peak_gpu_memory_mb",
    "trainable_parameter_count",
    "checkpoint_size_bytes",
)


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Bootstrap interval requires at least one value")
    if samples < 1:
        raise ValueError("Bootstrap samples must be at least 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence level must be between 0 and 1")
    if array.size == 1:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def load_study_records(
    manifest_path: str | Path,
    *,
    expected_dataset: str,
    expected_requests: Sequence[str],
    expected_seeds: Sequence[int],
    required_methods: Sequence[str],
    required_metrics: Sequence[str],
) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    manifest = pd.read_csv(manifest_path)
    required_manifest = {"request", "seed", "comparison_path"}
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_manifest:
        raise ValueError(
            f"Study manifest missing columns: {sorted(missing_manifest)}"
        )
    if manifest.duplicated(["request", "seed"]).any():
        raise ValueError("Study manifest contains duplicate request/seed rows")

    expected_pairs = {
        (str(request), int(seed))
        for request in expected_requests
        for seed in expected_seeds
    }
    actual_pairs = {
        (str(row.request), int(row.seed))
        for row in manifest.itertuples(index=False)
    }
    if actual_pairs != expected_pairs:
        raise ValueError(
            "Study manifest request/seed matrix mismatch: "
            f"missing={sorted(expected_pairs - actual_pairs)}, "
            f"unexpected={sorted(actual_pairs - expected_pairs)}"
        )

    frames = []
    for row in manifest.itertuples(index=False):
        comparison_path = Path(str(row.comparison_path))
        if not comparison_path.is_absolute():
            comparison_path = manifest_path.parent / comparison_path
        if not comparison_path.is_file():
            raise FileNotFoundError(
                f"Comparison file not found: {comparison_path}"
            )
        comparison = pd.read_csv(comparison_path)
        required_columns = {
            "model",
            *PROVENANCE_COLUMNS,
            *required_metrics,
        }
        missing = required_columns - set(comparison.columns)
        if missing:
            raise ValueError(
                f"{comparison_path} missing columns: {sorted(missing)}"
            )
        if comparison["model"].duplicated().any():
            raise ValueError(
                f"{comparison_path} contains duplicate model rows"
            )
        methods = set(comparison["model"].astype(str))
        missing_methods = set(required_methods) - methods
        if missing_methods:
            raise ValueError(
                f"{comparison_path} missing methods: {sorted(missing_methods)}"
            )
        selected = comparison[
            comparison["model"].isin(required_methods)
        ].copy()
        if set(selected["dataset"].dropna().astype(str)) != {
            expected_dataset
        }:
            raise ValueError(
                f"{comparison_path} does not contain dataset={expected_dataset}"
            )
        if set(selected["request"].dropna().astype(str)) != {
            str(row.request)
        }:
            raise ValueError(
                f"{comparison_path} request does not match {row.request}"
            )
        numeric_seeds = pd.to_numeric(selected["seed"], errors="coerce")
        if set(numeric_seeds.dropna().astype(int)) != {int(row.seed)}:
            raise ValueError(
                f"{comparison_path} seed does not match {row.seed}"
            )
        selected["comparison_path"] = str(comparison_path.resolve())
        frames.append(selected)

    records = pd.concat(frames, ignore_index=True)
    if records.duplicated(["request", "seed", "model"]).any():
        raise ValueError("Study records contain duplicate request/seed/model rows")
    missing_provenance = {
        column: int(records[column].isna().sum())
        for column in PROVENANCE_COLUMNS
        if records[column].isna().any()
    }
    if missing_provenance:
        raise ValueError(
            f"Study records contain missing provenance: {missing_provenance}"
        )
    git_commits = set(records["git_commit"].astype(str))
    if len(git_commits) != 1:
        raise ValueError(f"Study mixes Git commits: {sorted(git_commits)}")
    gpu_names = set(records["gpu_name"].astype(str))
    if len(gpu_names) != 1:
        raise ValueError(f"Study mixes GPU models: {sorted(gpu_names)}")

    for request in expected_requests:
        request_rows = records[records["request"] == request]
        for metric in required_metrics:
            counts = request_rows.groupby("model")[metric].count()
            nonzero = counts[counts > 0]
            if not nonzero.empty and set(nonzero.tolist()) != {
                len(expected_seeds)
            }:
                raise ValueError(
                    f"Metric {metric} is incomplete for request={request}: "
                    f"{counts.to_dict()}"
                )
    return records


def aggregate_study(
    records: pd.DataFrame,
    *,
    metrics: Sequence[str],
    bootstrap_samples: int,
    confidence: float,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = records.groupby(["dataset", "request", "model"], sort=True)
    for (dataset, request, model), group in grouped:
        for metric_index, metric in enumerate(metrics):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                rows.append(
                    {
                        "dataset": dataset,
                        "request": request,
                        "model": model,
                        "metric": metric,
                        "n": 0,
                        "mean": None,
                        "std": None,
                        "ci_low": None,
                        "ci_high": None,
                    }
                )
                continue
            ci_low, ci_high = bootstrap_mean_interval(
                values.to_numpy(),
                samples=bootstrap_samples,
                confidence=confidence,
                seed=bootstrap_seed + metric_index,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "request": request,
                    "model": model,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "std": (
                        float(values.std(ddof=1))
                        if values.size > 1
                        else 0.0
                    ),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def paired_study_differences(
    records: pd.DataFrame,
    *,
    primary_method: str,
    baseline_methods: Sequence[str],
    metrics: Sequence[str],
    bootstrap_samples: int,
    confidence: float,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for request, request_rows in records.groupby("request", sort=True):
        primary = request_rows[
            request_rows["model"] == primary_method
        ].set_index("seed")
        if primary.empty:
            raise ValueError(
                f"Primary method {primary_method!r} missing for {request}"
            )
        for baseline_index, baseline_method in enumerate(baseline_methods):
            baseline = request_rows[
                request_rows["model"] == baseline_method
            ].set_index("seed")
            if baseline.empty:
                raise ValueError(
                    f"Baseline {baseline_method!r} missing for {request}"
                )
            if set(primary.index) != set(baseline.index):
                raise ValueError(
                    f"Seed mismatch for {primary_method}/{baseline_method} "
                    f"on {request}"
                )
            for metric_index, metric in enumerate(metrics):
                paired = pd.concat(
                    [
                        pd.to_numeric(primary[metric], errors="coerce").rename(
                            "primary"
                        ),
                        pd.to_numeric(baseline[metric], errors="coerce").rename(
                            "baseline"
                        ),
                    ],
                    axis=1,
                ).dropna()
                if paired.empty:
                    continue
                delta = paired["primary"] - paired["baseline"]
                ci_low, ci_high = bootstrap_mean_interval(
                    delta.to_numpy(),
                    samples=bootstrap_samples,
                    confidence=confidence,
                    seed=(
                        bootstrap_seed
                        + 10_000
                        + baseline_index * len(metrics)
                        + metric_index
                    ),
                )
                rows.append(
                    {
                        "dataset": request_rows["dataset"].iloc[0],
                        "request": request,
                        "primary_method": primary_method,
                        "baseline_method": baseline_method,
                        "metric": metric,
                        "n": int(delta.size),
                        "mean_delta": float(delta.mean()),
                        "std_delta": (
                            float(delta.std(ddof=1))
                            if delta.size > 1
                            else 0.0
                        ),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )
    return pd.DataFrame(rows)


def write_study_artifacts(
    records: pd.DataFrame,
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    output_dir: str | Path,
    validation: dict[str, Any],
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "records_csv": output_dir / "study_records.csv",
        "aggregate_csv": output_dir / "study_aggregate.csv",
        "aggregate_json": output_dir / "study_aggregate.json",
        "aggregate_markdown": output_dir / "study_aggregate.md",
        "aggregate_latex": output_dir / "study_aggregate.tex",
        "paired_csv": output_dir / "study_paired_differences.csv",
        "paired_json": output_dir / "study_paired_differences.json",
        "paired_markdown": output_dir / "study_paired_differences.md",
        "paired_latex": output_dir / "study_paired_differences.tex",
        "validation_json": output_dir / "study_validation.json",
    }
    records.to_csv(paths["records_csv"], index=False)
    aggregate.to_csv(paths["aggregate_csv"], index=False)
    aggregate.to_json(paths["aggregate_json"], orient="records", indent=2)
    paths["aggregate_markdown"].write_text(
        "# Multi-Seed Aggregate Results\n\n"
        + aggregate.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    paths["aggregate_latex"].write_text(
        aggregate.to_latex(
            index=False,
            longtable=True,
            float_format="%.4f",
            caption="Multi-seed aggregate results with bootstrap intervals.",
            label="tab:study_aggregate",
        ),
        encoding="utf-8",
    )
    paired.to_csv(paths["paired_csv"], index=False)
    paired.to_json(paths["paired_json"], orient="records", indent=2)
    paths["paired_markdown"].write_text(
        "# Paired Per-Seed Differences\n\n"
        "Positive deltas mean primary method minus baseline.\n\n"
        + paired.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    paths["paired_latex"].write_text(
        paired.to_latex(
            index=False,
            longtable=True,
            float_format="%.4f",
            caption=(
                "Paired per-seed differences: primary method minus baseline."
            ),
            label="tab:study_paired",
        ),
        encoding="utf-8",
    )
    paths["validation_json"].write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    return {name: str(path) for name, path in paths.items()}
