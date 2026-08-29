from __future__ import annotations

from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVATION_SCRIPT = REPO_ROOT / "env_activation.sh"


def _base_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    scratch_root = tmp_path / "scratch"
    project = scratch_root / "machine-unlearning"
    bin_dir = project / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    uv = bin_dir / "uv"
    uv.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    uv.chmod(0o755)

    environment = {
        "HOME": str(tmp_path / "home"),
        "PATH": "/usr/bin:/bin",
        "SCRATCH": str(scratch_root),
        "USER": "activation-test-user",
    }
    return environment, project


def test_activation_uses_scratch_environment_without_manual_exports(
    tmp_path: Path,
) -> None:
    environment, project = _base_environment(tmp_path)
    activate = project / ".venv" / "bin" / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(
        f'export VIRTUAL_ENV="{project / ".venv"}"\n', encoding="utf-8"
    )

    command = (
        f'source "{ACTIVATION_SCRIPT}" && '
        "printf '%s\\n' \"$SCRATCH_PROJECT\" \"$UV_PROJECT_ENVIRONMENT\" "
        '"$UV_CACHE_DIR" "$HF_HOME"'
    )
    result = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[-4:] == [
        str(project),
        str(project / ".venv"),
        str(project / ".uv-cache"),
        str(project / "huggingface_cache"),
    ]


def test_missing_environment_still_exports_scratch_uv_cache(
    tmp_path: Path,
) -> None:
    environment, project = _base_environment(tmp_path)
    command = (
        f'source "{ACTIVATION_SCRIPT}" || true; '
        "printf '%s\\n' \"$UV_PROJECT_ENVIRONMENT\" \"$UV_CACHE_DIR\""
    )
    result = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0
    assert result.stdout.strip().splitlines() == [
        str(project / ".venv"),
        str(project / ".uv-cache"),
    ]
    assert "uv environment not found" in result.stderr


def test_activation_discovers_user_scratch_when_scratch_is_not_exported(
    tmp_path: Path,
) -> None:
    user = "activation-test-user"
    scratch_base = tmp_path / "system-scratch"
    scratch_root = scratch_base / user
    project = scratch_root / "machine-unlearning"
    bin_dir = project / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    uv = bin_dir / "uv"
    uv.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    uv.chmod(0o755)
    activate = project / ".venv" / "bin" / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(
        f'export VIRTUAL_ENV="{project / ".venv"}"\n', encoding="utf-8"
    )
    environment = {
        "HOME": str(tmp_path / "home"),
        "PATH": "/usr/bin:/bin",
        "USER": user,
        "UNML_SCRATCH_BASE": str(scratch_base),
    }
    command = (
        f'source "{ACTIVATION_SCRIPT}" && '
        "printf '%s\n' \"$SCRATCH\" \"$SCRATCH_PROJECT\" "
        '"$UV_PROJECT_ENVIRONMENT" "$UNML_OUTPUTS"'
    )

    result = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-4:] == [
        str(scratch_root),
        str(project),
        str(project / ".venv"),
        str(project / "outputs"),
    ]
