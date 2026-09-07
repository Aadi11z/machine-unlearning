from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "src/hpc/submit_canonical_oracle.slurm"


@pytest.fixture
def cluster(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """Stub cluster activation and commands, without touching real scratch."""
    root = tmp_path / "checkout with spaces"
    root.mkdir()
    (root / "env_activation.sh").write_text(
        'export SCRATCH_PROJECT="$SCRATCH/machine-unlearning"\n'
        'export UNML_OUTPUTS="$UNML_ROOT/outputs"\n'
        'export UNML_DATA="$UNML_ROOT/data"\n'
        'export HF_HOME="$UNML_ROOT/cache"\n'
        'export UV_PROJECT_ENVIRONMENT="$UNML_ROOT/.venv"\n',
        encoding="utf-8",
    )
    environment = {
        "PATH": os.defpath,
        "USER": "oracle-test",
        "SLURM_JOB_ID": "123",
        "SLURM_SUBMIT_DIR": str(root),
        "UNML_RUN_ID": "test-run",
        "UNML_ORACLE_REQUEST": "flowers_superclass",
        "CALL_LOG": str(tmp_path / "calls"),
    }
    return environment, root


def run_wrapper(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash", "-c",
            'spack() { return 0; }; '
            'uv() { printf "%s\\n" "$*" >> "$CALL_LOG"; '
            'return "${UV_STUB_STATUS:-0}"; }; '
            'source "$1"',
            "test-oracle", str(WRAPPER),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_oracle_wrapper_resolves_migrated_paths(cluster) -> None:
    environment, root = cluster
    result = run_wrapper(environment)
    assert result.returncode == 0, result.stderr
    assert "scratch_project=/scratch/oracle-test/machine-unlearning" in result.stdout
    assert f"root={root} data={root}/data cache={root}/cache" in result.stdout
    oracle = root / "outputs/cifar100/oracle/test-run/flowers_superclass"
    baseline = root / "outputs/cifar100/baseline"
    calls = Path(environment["CALL_LOG"]).read_text().splitlines()
    assert len(calls) == 6
    assert all(call.startswith("run --locked python ") for call in calls)
    assert str(baseline) in calls[0]
    assert f"--output-path {oracle}/splits/flowers_superclass.json" in calls[2]
    assert f"--initial-checkpoint {baseline}/checkpoints/base_init.pt" in calls[3]
    assert f"--source-metrics {baseline}/metrics/finetune_metrics.json" in calls[3]
    assert f"--output-dir {oracle}" in calls[3]
    assert "evaluate_canonical_reference_pair.py" in calls[4]
    assert "write_retraining_oracle_manifest.py" in calls[5]


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("SLURM_JOB_ID", "", "Submit this file with sbatch"),
        ("UNML_RUN_ID", "../escape", "Run id and request must"),
        ("UNML_ORACLE_REQUEST", "/tmp/escape", "Run id and request must"),
        ("UNML_RUN_ID", "", "Set a new oracle run id"),
        ("UNML_ORACLE_REQUEST", "", "Set UNML_ORACLE_REQUEST"),
    ],
)
def test_oracle_wrapper_rejects_invalid_submission(cluster, key, value, message) -> None:
    environment, _ = cluster
    environment[key] = value
    result = run_wrapper(environment)
    assert result.returncode != 0
    assert message in result.stderr
    assert not Path(environment["CALL_LOG"]).exists()


def test_oracle_wrapper_refuses_existing_run(cluster) -> None:
    environment, root = cluster
    (root / "outputs/cifar100/oracle/test-run/flowers_superclass").mkdir(parents=True)
    result = run_wrapper(environment)
    assert result.returncode == 2
    assert "Oracle run destination already exists" in result.stderr
    assert not Path(environment["CALL_LOG"]).exists()


def test_oracle_wrapper_stops_after_failed_preflight(cluster) -> None:
    environment, _ = cluster
    environment["UV_STUB_STATUS"] = "1"
    result = run_wrapper(environment)
    assert result.returncode == 1
    calls = Path(environment["CALL_LOG"]).read_text().splitlines()
    assert len(calls) == 1
    assert "verify_manifest_artifacts" in calls[0]
