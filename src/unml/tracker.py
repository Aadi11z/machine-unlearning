"""Experiment tracker that appends training and unlearning rows to Markdown."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence


def _project_root() -> Path:
    path = Path(__file__).resolve().parent
    for _ in range(5):
        if (path / "pyproject.toml").exists():
            return path
        path = path.parent
    return Path(__file__).resolve().parent.parent.parent


def _fmt(value: Any, precision: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def _ensure_header(path: Path, title: str, columns: Sequence[str]) -> None:
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
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row + "\n")


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


def log_finetune_epoch(
    cfg_dict: Dict[str, Any],
    epoch: int,
    train_loss: float,
    eval_metrics: Dict[str, float],
) -> None:
    path = _project_root() / "FINETUNE_TRACKER.md"
    _ensure_header(path, "Finetuning Experiment Tracker", _FT_COLUMNS)
    _append_row(
        path,
        [
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
        ],
    )


def log_finetune_summary(
    cfg_dict: Dict[str, Any],
    best_retain_val_acc: float,
    total_epochs: int,
) -> None:
    path = _project_root() / "FINETUNE_TRACKER.md"
    if not path.exists():
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n> **Run complete** | best_retain_val_acc={best_retain_val_acc:.4f} | "
            f"epochs={total_epochs} | rank={cfg_dict.get('adapter_rank')} | "
            f"lr={cfg_dict.get('lr')} | {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        )


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

_UL_PARTS_IDX: Dict[str, int] = {col: i + 1 for i, col in enumerate(_UL_COLUMNS)}


def log_unlearn_run(cfg_dict: Dict[str, Any], metrics: Dict[str, float]) -> None:
    path = _project_root() / "UNLEARN_EVAL_TRACKER.md"
    _ensure_header(path, "Unlearning & Evaluation Experiment Tracker", _UL_COLUMNS)
    _append_row(
        path,
        [
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
        ],
    )


def update_unlearn_with_attacks(
    method: str,
    attack_metrics: Dict[str, float],
) -> None:
    path = _project_root() / "UNLEARN_EVAL_TRACKER.md"
    if not path.exists():
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    for index in range(len(lines) - 1, -1, -1):
        parts = [cell.strip() for cell in lines[index].split("|")]
        if len(parts) <= 2 or parts[2] != method:
            continue
        if len(parts) >= len(_UL_COLUMNS) + 2:
            parts[_UL_PARTS_IDX["forget_quality"]] = _fmt(
                attack_metrics.get("forget_quality", 0.0)
            )
            parts[_UL_PARTS_IDX["mia_auc_conf"]] = _fmt(
                attack_metrics.get("mia_auc_confidence", 0.0)
            )
            parts[_UL_PARTS_IDX["mia_auc_delta"]] = _fmt(
                attack_metrics.get("mia_auc_delta", 0.0)
            )
            parts[_UL_PARTS_IDX["forget_drop"]] = _fmt(
                attack_metrics.get("forget_drop", 0.0)
            )
            lines[index] = "| " + " | ".join(parts[1:-1]) + " |"
        break

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
