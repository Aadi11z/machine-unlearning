from __future__ import annotations
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping
import numpy as np
import torch

DEFAULT_PROMPT_TEMPLATE = "a photo of a {}"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)

def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}

def tensor_to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)

def format_metrics(metrics: Mapping[str, float], precision: int = 4) -> str:
    parts = [f"{k}={v:.{precision}f}" for k, v in sorted(metrics.items())]
    return ", ".join(parts)


def environment_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def transformers_offline() -> bool:
    return environment_flag("HF_HUB_OFFLINE") or environment_flag(
        "TRANSFORMERS_OFFLINE"
    )


def git_commit(repo_root: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def collect_run_provenance(
    device: torch.device,
    repo_root: str | Path | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "git_commit": git_commit(repo_root),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": str(device),
        "transformers_offline": transformers_offline(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        payload.update(
            {
                "cuda_version": torch.version.cuda,
                "gpu_name": properties.name,
                "gpu_total_memory_mb": properties.total_memory / (1024**2),
            }
        )
    return payload
