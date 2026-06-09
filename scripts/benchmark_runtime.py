#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from unml.config import (
    load_runtime_config,
    resolve_dataset_and_split_path,
    resolve_request_name,
    resolve_stage_output_dir,
)


VARIANTS: dict[str, list[str]] = {
    "configured": [],
    "no_checkpointing": ["--no-gradient-checkpointing"],
    "synchronous_input": [
        "--no-pin-memory",
        "--no-persistent-workers",
        "--no-non-blocking",
    ],
    "workers8": ["--num-workers", "8"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and summarize isolated fine-tuning smoke ablations"
    )
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "config" / "parameters.yaml")
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--request", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(VARIANTS),
        help="Variant to run; repeat the option to select multiple variants",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeated measurements per variant; use at least 3 for reporting",
    )
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Run variants even when a complete metrics file already exists",
    )
    parser.add_argument(
        "--show-commands",
        action="store_true",
        help="Print the resolved benchmark commands without running them",
    )
    return parser.parse_args()


def build_train_command(
    *,
    config: str,
    dataset: str,
    request: str | None,
    output_dir: Path,
    device: str,
    offline: bool,
    variant: str,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_vlm.py"),
        "--config",
        config,
        "--dataset",
        dataset,
        "--output-dir",
        str(output_dir),
        "--smoke",
        "--device",
        device,
    ]
    if request:
        command.extend(["--request", request])
    if offline:
        command.append("--offline")
    command.extend(VARIANTS[variant])
    return command


def read_record(
    variant: str, repeat: int, metrics_path: Path
) -> dict[str, Any]:
    with metrics_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not payload.get("smoke_mode"):
        raise ValueError(f"Benchmark input is not a smoke run: {metrics_path}")
    if not payload.get("evaluation_is_partial"):
        raise ValueError(f"Benchmark evaluation is not bounded: {metrics_path}")
    if not payload.get("checkpoint_payload_verified"):
        raise ValueError(f"Checkpoint verification failed: {metrics_path}")
    if payload.get("gradient_parameter_count", 0) < 1:
        raise ValueError(f"No trainable gradient was recorded: {metrics_path}")
    benchmark = payload["benchmark"]
    config = payload["config"]
    provenance = payload["provenance"]
    required_timing = {
        "optimization_seconds",
        "evaluation_seconds",
        "optimization_examples_per_second",
    }
    missing_timing = required_timing - benchmark.keys()
    if missing_timing:
        raise ValueError(
            f"Benchmark metrics predate component timing "
            f"({sorted(missing_timing)}): {metrics_path}"
        )
    return {
        "variant": variant,
        "repeat": repeat,
        "git_commit": provenance.get("git_commit"),
        "gpu_name": provenance.get("gpu_name"),
        "cuda_version": provenance.get("cuda_version"),
        "dataset": payload["dataset"],
        "model_name": config["model_name"],
        "adapter_type": config["adapter_type"],
        "precision": config["precision"],
        "seed": config["seed"],
        "global_steps": payload["global_steps"],
        "processed_examples": benchmark["processed_examples"],
        "setup_seconds": benchmark["setup_seconds"],
        "optimization_seconds": benchmark["optimization_seconds"],
        "evaluation_seconds": benchmark["evaluation_seconds"],
        "total_seconds": benchmark["total_seconds"],
        "optimization_examples_per_second": benchmark[
            "optimization_examples_per_second"
        ],
        "peak_gpu_memory_mb": benchmark["peak_gpu_memory_mb"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "physical_batch_size": benchmark["physical_batch_size"],
        "effective_batch_size": benchmark["effective_batch_size"],
        "num_workers": benchmark["num_workers"],
        "pin_memory": benchmark["pin_memory"],
        "persistent_workers": benchmark["persistent_workers"],
        "prefetch_factor": benchmark["prefetch_factor"],
        "non_blocking": benchmark["non_blocking"],
        "metrics_path": str(metrics_path),
    }


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["variant"], []).append(record)
    summaries = []
    for variant, rows in grouped.items():
        throughputs = [
            float(row["optimization_examples_per_second"]) for row in rows
        ]
        memories = [float(row["peak_gpu_memory_mb"]) for row in rows]
        summaries.append(
            {
                "variant": variant,
                "repeats": len(rows),
                "optimization_examples_per_second_mean": statistics.mean(
                    throughputs
                ),
                "optimization_examples_per_second_std": (
                    statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0
                ),
                "peak_gpu_memory_mb_mean": statistics.mean(memories),
                "peak_gpu_memory_mb_std": (
                    statistics.stdev(memories) if len(memories) > 1 else 0.0
                ),
                "gradient_checkpointing": rows[0]["gradient_checkpointing"],
                "num_workers": rows[0]["num_workers"],
                "pin_memory": rows[0]["pin_memory"],
                "persistent_workers": rows[0]["persistent_workers"],
                "prefetch_factor": rows[0]["prefetch_factor"],
                "non_blocking": rows[0]["non_blocking"],
            }
        )
    configured = next(
        (row for row in summaries if row["variant"] == "configured"), None
    )
    baseline = (
        configured["optimization_examples_per_second_mean"]
        if configured
        else None
    )
    for row in summaries:
        row["throughput_vs_configured"] = (
            row["optimization_examples_per_second_mean"] / baseline
            if baseline
            else None
        )
    return summaries


def validate_records(records: list[dict[str, Any]]) -> None:
    invariant_keys = (
        "git_commit",
        "gpu_name",
        "cuda_version",
        "dataset",
        "model_name",
        "adapter_type",
        "precision",
        "seed",
        "global_steps",
        "processed_examples",
        "physical_batch_size",
        "effective_batch_size",
    )
    for key in invariant_keys:
        values = {record[key] for record in records}
        if len(values) > 1:
            raise ValueError(
                f"Benchmark records do not share one {key}: "
                f"{sorted(repr(value) for value in values)}"
            )


def write_summary(records: list[dict[str, Any]], output_dir: Path) -> None:
    validate_records(records)
    summaries = aggregate_records(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "runtime_benchmark_raw.json"
    csv_path = output_dir / "runtime_benchmark_raw.csv"
    summary_json_path = output_dir / "runtime_benchmark_summary.json"
    summary_csv_path = output_dir / "runtime_benchmark_summary.csv"
    markdown_path = output_dir / "runtime_benchmark.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    fieldnames = list(records[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# Runtime Smoke Benchmark\n\n")
        handle.write(
            "| Variant | Runs | Opt. examples/s | Relative | Peak MiB | "
            "Checkpointing | Workers | Pinned | Persistent | Nonblocking |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---|---:|---|---|---|\n")
        for record in summaries:
            relative = record["throughput_vs_configured"]
            relative_text = f"{relative:.3f}" if relative is not None else "n/a"
            handle.write(
                f"| {record['variant']} "
                f"| {record['repeats']} "
                f"| {record['optimization_examples_per_second_mean']:.2f} "
                f"+/- {record['optimization_examples_per_second_std']:.2f} "
                f"| {relative_text} "
                f"| {record['peak_gpu_memory_mb_mean']:.1f} "
                f"+/- {record['peak_gpu_memory_mb_std']:.1f} "
                f"| {record['gradient_checkpointing']} "
                f"| {record['num_workers']} "
                f"| {record['pin_memory']} "
                f"| {record['persistent_workers']} "
                f"| {record['non_blocking']} |\n"
            )


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    runtime_cfg = load_runtime_config(args.config)
    dataset, _ = resolve_dataset_and_split_path(
        args.dataset, None, runtime_cfg, args.request
    )
    request = resolve_request_name(args.request, runtime_cfg, dataset)
    default_training_dir = resolve_stage_output_dir(
        None, runtime_cfg, dataset, "training", request
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_training_dir.parent / "runtime_benchmark"
    )
    variants = args.variant or list(VARIANTS)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    records = []
    for variant in variants:
        for repeat in range(1, args.repeats + 1):
            variant_dir = output_dir / variant / f"repeat_{repeat:02d}"
            metrics_path = variant_dir / "metrics" / "finetune_metrics.json"
            command = build_train_command(
                config=args.config,
                dataset=dataset,
                request=request,
                output_dir=variant_dir,
                device=args.device,
                offline=args.offline,
                variant=variant,
            )
            if args.show_commands:
                print("[benchmark]", " ".join(command), flush=True)
                continue
            if args.rerun or not metrics_path.exists():
                print("[benchmark]", " ".join(command), flush=True)
                subprocess.run(command, check=True, env=env, cwd=REPO_ROOT)
            else:
                print(f"[benchmark] reusing {metrics_path}", flush=True)
            records.append(read_record(variant, repeat, metrics_path))

    if args.show_commands:
        return
    write_summary(records, output_dir)
    print(f"[benchmark] summary={output_dir / 'runtime_benchmark.md'}")


if __name__ == "__main__":
    main()
