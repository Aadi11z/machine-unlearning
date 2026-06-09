from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS = [
    "scripts/cache_model.py",
    "scripts/benchmark_runtime.py",
    "scripts/probe_checkpoint.py",
    "scripts/prepare_data.py",
    "scripts/train_vlm.py",
    "scripts/run_unlearning.py",
    "scripts/evaluate_attacks.py",
    "scripts/run_pipeline.py",
]


@pytest.mark.parametrize("script_path", SCRIPTS)
def test_script_help_runs(script_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(repo_root / script_path), "--help"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert proc.returncode == 0
    assert "usage:" in proc.stdout.lower()


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


def test_runtime_benchmark_show_commands_is_non_executing() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "scripts/benchmark_runtime.py"),
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
