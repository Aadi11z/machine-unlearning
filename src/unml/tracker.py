"""Experiment tracker that appends results to markdown files.

Two tracker files are maintained at the project root:
- FINETUNE_TRACKER.md  : per-epoch metrics for each finetuning run
- UNLEARN_EVAL_TRACKER.md : unlearning + attack evaluation metrics
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _project_root() -> Path:
    """Walk up from this file to find the repo root (contains pyproject.toml)."""
    p = Path(__file__).resolve().parent
    for _ in range(5):
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    # Fallback: two levels up from src/unml/
    return Path(__file__).resolve().parent.parent.parent


def _fmt(v: Any, precision: int = 4) -> str:
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return str(v)


def _ensure_header(path: Path, title: str, columns: Sequence[str]) -> None:
    """Create the file with header if it doesn't exist or is empty."""
    if path.exists() and path.stat().st_size > 0:
        return
    header_row = "| " + " | ".join(columns) + " |"
    sep_row = "| " + " | ".join("---" for _ in columns) + " |"
    path.write_text(
        f"# {title}\n\n"
        f"Auto-updated by the training pipeline. Each row = one run/epoch.\n\n"
        f"{header_row}\n{sep_row}\n",
        encoding="utf-8",
    )


def _append_row(path: Path, values: Sequence[str]) -> None:
    row = "| " + " | ".join(values) + " |"
    with open(path, "a", encoding="utf-8") as f:
        f.write(row + "\n")


# ---------------------------------------------------------------------------
# Finetuning tracker
# ---------------------------------------------------------------------------

_FT_COLUMNS = [
    "timestamp",
    "epoch",
    "adapter_rank",
    "adapter_alpha",
    "lr",
    "weight_decay",
    "batch_size",
    "seed",
    "train_loss",
    "retain_val_acc",
    "retain_val_loss",
    "test_all_acc",
    "test_retain_acc",
    "forget_train_acc",
]


def log_finetune_epoch(cfg_dict: Dict[str, Any], epoch: int, train_loss: float, eval_metrics: Dict[str, float]) -> None:
    root = _project_root()
    path = root / "FINETUNE_TRACKER.md"
    _ensure_header(path, "Finetuning Experiment Tracker", _FT_COLUMNS)

    values = [
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        str(epoch),
        str(cfg_dict.get("adapter_rank", "")),
        _fmt(cfg_dict.get("adapter_alpha", "")),
        _fmt(cfg_dict.get("lr", "")),
        _fmt(cfg_dict.get("weight_decay", "")),
        str(cfg_dict.get("batch_size", "")),
        str(cfg_dict.get("seed", "")),
        _fmt(train_loss),
        _fmt(eval_metrics.get("retain_val_acc", -0.0)),
        _fmt(eval_metrics.get("retain_val_loss", -0.0)),
        _fmt(eval_metrics.get("test_all_acc", -0.0)),
        _fmt(eval_metrics.get("test_retain_acc", -0.0)),
        _fmt(eval_metrics.get("forget_train_acc", -0.0)),
    ]
    _append_row(path, values)


def log_finetune_summary(cfg_dict: Dict[str, Any], best_retain_val_acc: float, total_epochs: int) -> None:
    root = _project_root()
    path = root / "FINETUNE_TRACKER.md"
    if not path.exists():
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            f"\n> **Run complete** | best_retain_val_acc={best_retain_val_acc:.4f} | "
            f"epochs={total_epochs} | rank={cfg_dict.get('adapter_rank')} | "
            f"lr={cfg_dict.get('lr')} | {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        )


# ---------------------------------------------------------------------------
# Unlearning + evaluation tracker
# ---------------------------------------------------------------------------

_UL_COLUMNS = [
    "timestamp",
    "method",
    "steps",
    "lr",
    "kl_temp",
    "kl_weight",
    "ga_weight",
    "cf_weight",
    "margin_weight",
    "margin",
    "seed",
    "retain_val_acc",
    "forget_acc",
    "test_retain_acc",
    "test_all_acc",
    "avg_train_loss",
    "forget_quality",
    "mia_auc_conf",
    "mia_auc_delta",
    "forget_drop",
]

# parts[0] is empty (before the first "|"), so column k maps to parts[k+1]
_UL_PARTS_IDX: Dict[str, int] = {col: i + 1 for i, col in enumerate(_UL_COLUMNS)}


def log_unlearn_run(cfg_dict: Dict[str, Any], metrics: Dict[str, float]) -> None:
    root = _project_root()
    path = root / "UNLEARN_EVAL_TRACKER.md"
    _ensure_header(path, "Unlearning & Evaluation Experiment Tracker", _UL_COLUMNS)

    values = [
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        str(cfg_dict.get("method", "")),
        str(cfg_dict.get("steps", "")),
        _fmt(cfg_dict.get("lr", "")),
        _fmt(cfg_dict.get("kl_temperature", "")),
        _fmt(cfg_dict.get("kl_weight", "")),
        _fmt(cfg_dict.get("ga_weight", "")),
        _fmt(cfg_dict.get("cf_weight", "")),
        _fmt(cfg_dict.get("margin_weight", "")),
        _fmt(cfg_dict.get("margin", "")),
        str(cfg_dict.get("seed", "")),
        _fmt(metrics.get("retain_val_acc", 0.0)),
        _fmt(metrics.get("forget_acc", 0.0)),
        _fmt(metrics.get("test_retain_acc", 0.0)),
        _fmt(metrics.get("test_all_acc", 0.0)),
        _fmt(metrics.get("avg_train_loss", 0.0)),
        _fmt(metrics.get("forget_quality", "")),
        _fmt(metrics.get("mia_auc_confidence", "")),
        _fmt(metrics.get("mia_auc_delta", "")),
        _fmt(metrics.get("forget_drop", "")),
    ]
    _append_row(path, values)


def update_unlearn_with_attacks(method: str, attack_metrics: Dict[str, float]) -> None:
    """Update the last row for a given method with attack evaluation metrics.

    Called from the attack comparison pipeline after MIA evaluation.
    This reads the tracker, finds the most recent row matching the method,
    and fills in the attack columns if they were empty.
    """
    root = _project_root()
    path = root / "c.md"
    if not path.exists():
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    # Find the last row matching this method and update attack columns
    for i in range(len(lines) - 1, -1, -1):
        parts = [c.strip() for c in lines[i].split("|")]
        # parts[0] is empty (before first |), parts[1]=timestamp, parts[2]=method
        if len(parts) > 2 and parts[2] == method:
            # Update attack columns — indices derived from _UL_COLUMNS to stay in sync
            if len(parts) >= len(_UL_COLUMNS) + 2:
                parts[_UL_PARTS_IDX["forget_quality"]] = _fmt(attack_metrics.get("forget_quality", 0.0))
                parts[_UL_PARTS_IDX["mia_auc_conf"]] = _fmt(attack_metrics.get("mia_auc_confidence", 0.0))
                parts[_UL_PARTS_IDX["mia_auc_delta"]] = _fmt(attack_metrics.get("mia_auc_delta", 0.0))
                parts[_UL_PARTS_IDX["forget_drop"]] = _fmt(attack_metrics.get("forget_drop", 0.0))
                lines[i] = " | ".join(parts)
            break

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
