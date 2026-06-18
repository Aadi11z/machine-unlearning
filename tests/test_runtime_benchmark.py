from __future__ import annotations

import json

import pytest

from helpers.benchmark_runtime import (
    aggregate_records,
    build_train_command,
    read_record,
    validate_records,
    write_summary,
)


def _record(
    variant: str,
    repeat: int,
    throughput: float,
    memory: float,
) -> dict:
    return {
        "variant": variant,
        "repeat": repeat,
        "git_commit": "abc123",
        "gpu_name": "A100",
        "cuda_version": "12.4",
        "dataset": "cifar100",
        "model_name": "openai/clip-vit-base-patch16",
        "adapter_type": "vision_lora",
        "precision": "bf16",
        "seed": 42,
        "global_steps": 50,
        "processed_examples": 6400,
        "setup_seconds": 2.0,
        "optimization_seconds": 10.0,
        "evaluation_seconds": 1.0,
        "total_seconds": 13.0,
        "optimization_examples_per_second": throughput,
        "peak_gpu_memory_mb": memory,
        "gradient_checkpointing": variant != "no_checkpointing",
        "physical_batch_size": 64,
        "effective_batch_size": 128,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "non_blocking": True,
        "metrics_path": f"{variant}/{repeat}.json",
    }


def test_build_train_command_applies_variant_overrides(tmp_path) -> None:
    command = build_train_command(
        config="config/parameters.yaml",
        dataset="cifar100",
        request="flowers_superclass",
        output_dir=tmp_path,
        device="cuda",
        offline=True,
        variant="synchronous_input",
    )

    assert "--smoke" in command
    assert "--offline" in command
    assert "--no-pin-memory" in command
    assert "--no-persistent-workers" in command
    assert "--no-non-blocking" in command


def test_aggregate_records_reports_mean_std_and_relative_throughput() -> None:
    records = [
        _record("configured", 1, 100.0, 1000.0),
        _record("configured", 2, 120.0, 1100.0),
        _record("workers8", 1, 132.0, 1200.0),
        _record("workers8", 2, 132.0, 1200.0),
    ]

    summaries = {row["variant"]: row for row in aggregate_records(records)}

    assert summaries["configured"][
        "optimization_examples_per_second_mean"
    ] == pytest.approx(110.0)
    assert summaries["configured"][
        "optimization_examples_per_second_std"
    ] > 0
    assert summaries["workers8"]["throughput_vs_configured"] == pytest.approx(
        1.2
    )


def test_validate_records_rejects_mixed_hardware() -> None:
    records = [
        _record("configured", 1, 100.0, 1000.0),
        _record("configured", 2, 100.0, 1000.0),
    ]
    records[1]["gpu_name"] = "H100"

    with pytest.raises(ValueError, match="gpu_name"):
        validate_records(records)


def test_read_record_requires_verified_current_smoke_metrics(tmp_path) -> None:
    metrics_path = tmp_path / "finetune_metrics.json"
    payload = {
        "smoke_mode": True,
        "evaluation_is_partial": True,
        "checkpoint_payload_verified": True,
        "gradient_parameter_count": 4,
        "dataset": "cifar100",
        "global_steps": 50,
        "benchmark": {
            "processed_examples": 6400,
            "setup_seconds": 2.0,
            "optimization_seconds": 10.0,
            "evaluation_seconds": 1.0,
            "total_seconds": 13.0,
            "optimization_examples_per_second": 640.0,
            "peak_gpu_memory_mb": 1000.0,
            "physical_batch_size": 64,
            "effective_batch_size": 128,
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "non_blocking": True,
        },
        "config": {
            "model_name": "openai/clip-vit-base-patch16",
            "adapter_type": "vision_lora",
            "precision": "bf16",
            "seed": 42,
            "gradient_checkpointing": True,
        },
        "provenance": {
            "git_commit": "abc123",
            "gpu_name": "A100",
            "cuda_version": "12.4",
        },
    }
    metrics_path.write_text(json.dumps(payload))

    record = read_record("configured", 1, metrics_path)

    assert record["dataset"] == "cifar100"
    assert record["optimization_examples_per_second"] == 640.0

    payload["gradient_parameter_count"] = 0
    metrics_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="gradient"):
        read_record("configured", 1, metrics_path)


def test_write_summary_creates_raw_and_aggregate_artifacts(tmp_path) -> None:
    records = [
        _record("configured", 1, 100.0, 1000.0),
        _record("configured", 2, 110.0, 1050.0),
    ]

    write_summary(records, tmp_path)

    assert (tmp_path / "runtime_benchmark_raw.csv").exists()
    assert (tmp_path / "runtime_benchmark_summary.csv").exists()
    assert (tmp_path / "runtime_benchmark.md").exists()
    summary = json.loads(
        (tmp_path / "runtime_benchmark_summary.json").read_text()
    )
    assert summary[0]["repeats"] == 2
