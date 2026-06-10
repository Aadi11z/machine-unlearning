#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from unml.config import load_runtime_config, resolve_str_list, resolve_value
from unml.study import (
    aggregate_study,
    load_study_records,
    paired_study_differences,
    write_study_artifacts,
)


DEFAULT_METRICS = (
    "target_test_acc",
    "sibling_test_macro_acc",
    "unrelated_test_macro_acc",
    "utility_test_all",
    "mia_auc_confidence",
    "mia_auc_loss",
    "oracle_prediction_kl_all",
    "oracle_embedding_cosine_all",
    "runtime_seconds",
    "peak_gpu_memory_mb",
    "trainable_parameter_count",
    "checkpoint_size_bytes",
)
DEFAULT_METHODS = (
    "finetuned",
    "retrain_oracle",
    "retain_only",
    "ga_kl",
    "counterfactual_rebind",
    "entropy_rebind",
    "h_tgsd",
    "h_tgsd_no_sibling_preservation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a multi-seed UNML study"
    )
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "config" / "parameters.yaml")
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--requests", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--primary-method", default=None)
    parser.add_argument("--baseline-methods", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    return parser.parse_args()


def _dataset_output_root(
    runtime_cfg: dict, dataset: str
) -> Path:
    environment_root = os.environ.get("UNML_OUTPUTS")
    if environment_root:
        return Path(environment_root) / dataset
    raw = str(runtime_cfg.get("outputs", {}).get("root", "outputs/{dataset}"))
    return Path(raw.format(dataset=dataset))


def main() -> None:
    args = parse_args()
    runtime_cfg = load_runtime_config(args.config)
    dataset = str(
        resolve_value(
            args.dataset, runtime_cfg, ("study", "dataset"), "cifar100"
        )
    )
    dataset_root = _dataset_output_root(runtime_cfg, dataset)
    manifest_path = Path(
        resolve_value(
            args.manifest,
            runtime_cfg,
            ("study", "manifest_path"),
            str(dataset_root / "study_manifest.csv"),
        )
    )
    output_dir = Path(
        resolve_value(
            args.output_dir,
            runtime_cfg,
            ("study", "output_dir"),
            str(dataset_root / "study_summary"),
        )
    )
    requests = resolve_str_list(
        args.requests,
        runtime_cfg,
        ("study", "requests"),
        ("flowers_superclass", "rose_selective"),
    )
    seeds = [
        int(seed)
        for seed in resolve_str_list(
            args.seeds,
            runtime_cfg,
            ("study", "seeds"),
            ("42", "123", "456"),
        )
    ]
    methods = resolve_str_list(
        args.methods,
        runtime_cfg,
        ("study", "methods"),
        DEFAULT_METHODS,
    )
    metrics = resolve_str_list(
        args.metrics,
        runtime_cfg,
        ("study", "metrics"),
        DEFAULT_METRICS,
    )
    primary_method = str(
        resolve_value(
            args.primary_method,
            runtime_cfg,
            ("study", "primary_method"),
            "h_tgsd",
        )
    )
    baseline_methods = resolve_str_list(
        args.baseline_methods,
        runtime_cfg,
        ("study", "baseline_methods"),
        (
            "retain_only",
            "ga_kl",
            "counterfactual_rebind",
            "entropy_rebind",
            "h_tgsd_no_sibling_preservation",
        ),
    )
    bootstrap_samples = int(
        resolve_value(
            args.bootstrap_samples,
            runtime_cfg,
            ("study", "bootstrap_samples"),
            10_000,
        )
    )
    confidence = float(
        resolve_value(
            args.confidence,
            runtime_cfg,
            ("study", "confidence"),
            0.95,
        )
    )
    bootstrap_seed = int(
        resolve_value(
            args.bootstrap_seed,
            runtime_cfg,
            ("study", "bootstrap_seed"),
            2026,
        )
    )

    records = load_study_records(
        manifest_path,
        expected_dataset=dataset,
        expected_requests=requests,
        expected_seeds=seeds,
        required_methods=methods,
        required_metrics=metrics,
    )
    aggregate = aggregate_study(
        records,
        metrics=metrics,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        bootstrap_seed=bootstrap_seed,
    )
    paired = paired_study_differences(
        records,
        primary_method=primary_method,
        baseline_methods=baseline_methods,
        metrics=metrics,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        bootstrap_seed=bootstrap_seed,
    )
    validation = {
        "status": "passed",
        "dataset": dataset,
        "requests": requests,
        "seeds": seeds,
        "methods": methods,
        "metrics": metrics,
        "primary_method": primary_method,
        "baseline_methods": baseline_methods,
        "bootstrap_samples": bootstrap_samples,
        "confidence": confidence,
        "bootstrap_seed": bootstrap_seed,
        "git_commit": records["git_commit"].iloc[0],
        "gpu_name": records["gpu_name"].iloc[0],
        "run_rows": len(records),
    }
    paths = write_study_artifacts(
        records, aggregate, paired, output_dir, validation
    )
    for name, path in paths.items():
        print(f"[study] {name}={path}", flush=True)


if __name__ == "__main__":
    main()
