from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from .utils import git_commit


def cache_manifest_path(model_name: str, hf_home: str | Path | None) -> Path:
    root = Path(
        hf_home
        if hf_home is not None
        else os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ).expanduser()
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "--", model_name).strip(".-")
    return root / "unml" / "cache_manifests" / f"{slug}.json"


def write_cache_manifest(
    *,
    model_name: str,
    dataset_name: str,
    hf_home: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    path = cache_manifest_path(model_name, hf_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset": dataset_name,
        "model_name": model_name,
        "hf_home": str(path.parents[2].resolve()),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repo_root),
        "components": [
            "CLIPImageProcessor",
            "CLIPTokenizer",
            "CLIPModel",
        ],
        "offline_reload_verified": True,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def validate_cache_manifest(
    *,
    model_name: str,
    dataset_name: str,
    hf_home: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    path = cache_manifest_path(model_name, hf_home)
    if not path.is_file():
        raise FileNotFoundError(
            f"Offline cache manifest not found: {path}. Run "
            f"`python helpers/cache_model.py --dataset {dataset_name}` "
            "on a network-enabled login node first."
        )
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {
        "schema_version": 1,
        "dataset": dataset_name,
        "model_name": model_name,
        "offline_reload_verified": True,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    manifest_hf_home = Path(str(payload.get("hf_home", ""))).expanduser()
    active_hf_home = path.parents[2].resolve()
    if manifest_hf_home.resolve() != active_hf_home:
        mismatches["hf_home"] = (
            str(manifest_hf_home),
            str(active_hf_home),
        )
    if mismatches:
        raise ValueError(f"Offline cache manifest mismatch: {mismatches}")
    return path, payload


def validate_virtual_environment(require_venv: bool) -> dict[str, Any]:
    active = sys.prefix != sys.base_prefix
    if require_venv and not active:
        raise RuntimeError(
            "No Python virtual environment is active. Source "
            "`env_activation.sh` before launching the job."
        )
    return {
        "active": active,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
    }


def validate_cuda(require_cuda: bool, precision: str) -> dict[str, Any]:
    available = torch.cuda.is_available()
    if require_cuda and not available:
        raise RuntimeError(
            "CUDA is required but unavailable to this process. Confirm the "
            "SLURM GPU allocation and that the venv contains a CUDA PyTorch build."
        )
    payload: dict[str, Any] = {
        "available": available,
        "torch_cuda_version": torch.version.cuda,
    }
    if available:
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        payload.update(
            {
                "device_index": device,
                "gpu_name": properties.name,
                "total_memory_mb": properties.total_memory / (1024**2),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
        if precision == "bf16" and not payload["bf16_supported"]:
            raise RuntimeError(
                f"Configured precision=bf16 is unsupported by {properties.name}"
            )
    return payload


def validate_writable_directory(path: str | Path) -> Path:
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".unml-preflight-", dir=directory, delete=True
    ):
        pass
    return directory.resolve()


def validate_scratch_layout(
    paths: Mapping[str, str | Path],
    scratch_project: str | Path | None,
) -> Path | None:
    if scratch_project is None:
        return None
    root = Path(scratch_project).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Scratch project directory not found: {root}")
    outside = {}
    for name, raw_path in paths.items():
        path = Path(raw_path).expanduser().resolve()
        if not path.is_relative_to(root):
            outside[name] = str(path)
    if outside:
        raise ValueError(
            f"Heavy artifact paths must be under scratch project {root}: "
            f"{outside}"
        )
    return root
