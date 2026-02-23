from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS = [
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
