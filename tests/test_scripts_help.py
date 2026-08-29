from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cli_smoke


SCRIPTS = [
    "helpers/cache_model.py",
    "helpers/preflight_run.py",
    "helpers/summarize_study.py",
    "scripts/run_study.py",
    "helpers/benchmark_runtime.py",
    "helpers/probe_checkpoint.py",
    "helpers/sweep_finetune.py",
    "scripts/prepare_data.py",
    "scripts/train_vlm.py",
    "scripts/run_unlearning.py",
    "scripts/evaluate_attacks.py",
    "scripts/run_pipeline.py",
    "scripts/run_lora_pilots.py",
    "scripts/generate_canonical_split.py",
    "scripts/train_canonical_baseline.py",
    "scripts/generate_canonical_oracle_split.py",
]


@pytest.mark.parametrize("script_path", SCRIPTS)
def test_script_help_runs(script_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(repo_root / script_path), "--help"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert proc.returncode == 0
    assert "usage:" in proc.stdout.lower()
    if script_path == "scripts/generate_canonical_split.py":
        assert "no forget request" in proc.stdout.lower()


def test_pipeline_show_config_does_not_start_pipeline() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "scripts/run_pipeline.py"),
        "--show-config",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False, cwd=repo_root
    )

    assert proc.returncode == 0
    assert "dataset=cifar10" in proc.stdout
    assert "output_root=outputs/cifar10" in proc.stdout
    assert "[cmd]" not in proc.stdout


def test_pipeline_cifar100_selects_vit_b16_vision_lora() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "scripts/run_pipeline.py"),
        "--dataset",
        "cifar100",
        "--show-config",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False, cwd=repo_root
    )

    assert proc.returncode == 0
    assert "model_name=openai/clip-vit-base-patch16" in proc.stdout
    assert "adapter_type=vision_lora" in proc.stdout
    assert "retraining.enabled=True" in proc.stdout
    assert (
        "retraining_dir=outputs/cifar100/flowers_superclass/retrain_oracle"
        in proc.stdout
    )


def test_runtime_benchmark_show_commands_is_non_executing() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "helpers/benchmark_runtime.py"),
        "--dataset",
        "cifar100",
        "--request",
        "flowers_superclass",
        "--variant",
        "configured",
        "--repeats",
        "2",
        "--show-commands",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False, cwd=repo_root
    )

    assert proc.returncode == 0
    assert proc.stdout.count("[benchmark]") == 2
    assert "repeat_01" in proc.stdout
    assert "repeat_02" in proc.stdout
    assert "--smoke" in proc.stdout
    assert "--offline" in proc.stdout


def test_study_show_commands_is_seed_and_request_isolated() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "scripts/run_study.py"),
        "--show-commands",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False, cwd=repo_root
    )

    assert proc.returncode == 0
    assert proc.stdout.count("[study]") == 6
    assert "flowers_superclass/seed_42" in proc.stdout
    assert "rose_selective/seed_456" in proc.stdout
    assert "--split-path" in proc.stdout
    assert "--output-root" in proc.stdout
