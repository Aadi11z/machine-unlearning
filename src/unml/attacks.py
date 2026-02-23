from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor, CLIPTokenizer

from .data import CIFAR10_CLASSES, build_loaders, build_text_inputs
from .evaluate import collect_true_class_confidences, evaluate_classification
from .model import load_checkpoint
from .utils import get_device


@dataclass
class AttackConfig:
    data_dir: str
    split_path: str
    model_name: str
    base_checkpoint: str
    candidate_checkpoints: Sequence[str]
    candidate_names: Sequence[str]
    output_dir: str
    prompt_template: str = "a photo of a {}"
    batch_size: int = 128
    num_workers: int = 4
    max_attack_samples: int = 4000
    device: str = "auto"


def _truncate_equal(member: np.ndarray, non_member: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(member), len(non_member))
    return member[:n], non_member[:n]


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    unique = np.unique(y_true)
    if unique.size < 2:
        return 0.5
    return float(roc_auc_score(y_true, scores))


def _mia_metrics(member_current: np.ndarray, nonmember_current: np.ndarray, member_base: np.ndarray, nonmember_base: np.ndarray) -> Dict[str, float]:
    member_current, nonmember_current = _truncate_equal(member_current, nonmember_current)
    member_base, nonmember_base = _truncate_equal(member_base, nonmember_base)

    y = np.concatenate([np.ones(len(member_current)), np.zeros(len(nonmember_current))])
    conf_scores = np.concatenate([member_current, nonmember_current])
    delta_scores = np.concatenate([member_current - member_base, nonmember_current - nonmember_base])

    auc_conf = _safe_auc(y, conf_scores)
    auc_delta = _safe_auc(y, delta_scores)

    mia_resistance_conf = max(0.0, 1.0 - abs(auc_conf - 0.5) * 2)
    mia_resistance_delta = max(0.0, 1.0 - abs(auc_delta - 0.5) * 2)

    return {
        "mia_auc_confidence": auc_conf,
        "mia_auc_delta": auc_delta,
        "mia_resistance_confidence": mia_resistance_conf,
        "mia_resistance_delta": mia_resistance_delta,
    }


def _evaluate_model(model, base_model, loaders: Dict[str, DataLoader], class_text_inputs: Dict[str, torch.Tensor], device: torch.device, max_attack_samples: int) -> Dict[str, float]:
    util_test_all = evaluate_classification(model, loaders["test_all"], class_text_inputs, device)
    util_test_retain = evaluate_classification(model, loaders["test_retain"], class_text_inputs, device)
    forget_train = evaluate_classification(model, loaders["forget"], class_text_inputs, device)

    member = collect_true_class_confidences(
        model,
        loaders["forget"],
        class_text_inputs,
        device,
        max_samples=max_attack_samples,
    )
    nonmember = collect_true_class_confidences(
        model,
        loaders["test_forget"],
        class_text_inputs,
        device,
        max_samples=max_attack_samples,
    )

    member_base = collect_true_class_confidences(
        base_model,
        loaders["forget"],
        class_text_inputs,
        device,
        max_samples=max_attack_samples,
    )
    nonmember_base = collect_true_class_confidences(
        base_model,
        loaders["test_forget"],
        class_text_inputs,
        device,
        max_samples=max_attack_samples,
    )

    attacks = _mia_metrics(
        member_current=member["scores"],
        nonmember_current=nonmember["scores"],
        member_base=member_base["scores"],
        nonmember_base=nonmember_base["scores"],
    )

    forget_drop = max(0.0, 1.0 - forget_train["accuracy"])
    forget_quality = (forget_drop + attacks["mia_resistance_confidence"] + attacks["mia_resistance_delta"]) / 3.0

    return {
        "utility_test_all": util_test_all["accuracy"],
        "utility_test_retain": util_test_retain["accuracy"],
        "forget_train_acc": forget_train["accuracy"],
        "forget_drop": forget_drop,
        "forget_quality": forget_quality,
        **attacks,
    }


def _plot_tradeoff(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["utility_test_retain"], df["forget_quality"], s=80)
    for _, row in df.iterrows():
        ax.annotate(row["model"], (row["utility_test_retain"], row["forget_quality"]), xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Utility (Retain Test Accuracy)")
    ax.set_ylabel("Forget Quality (higher is better)")
    ax.set_title("Utility vs Forget Quality")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_attack_comparison(cfg: AttackConfig) -> Dict[str, str]:
    device = get_device(cfg.device)

    image_processor = CLIPImageProcessor.from_pretrained(cfg.model_name)
    tokenizer = CLIPTokenizer.from_pretrained(cfg.model_name)
    class_text_inputs = build_text_inputs(
        tokenizer,
        class_names=CIFAR10_CLASSES,
        template=cfg.prompt_template,
    )
    class_text_inputs = {k: v.to(device) for k, v in class_text_inputs.items()}

    loaders = build_loaders(
        data_dir=cfg.data_dir,
        split_path=cfg.split_path,
        image_processor=image_processor,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    base_model, _ = load_checkpoint(cfg.base_checkpoint, map_location=device)
    base_model = base_model.to(device).eval()

    records: List[Dict[str, float | str]] = []
    for name, ckpt_path in zip(cfg.candidate_names, cfg.candidate_checkpoints):
        model, meta = load_checkpoint(ckpt_path, map_location=device)
        model = model.to(device).eval()

        metrics = _evaluate_model(
            model=model,
            base_model=base_model,
            loaders=loaders,
            class_text_inputs=class_text_inputs,
            device=device,
            max_attack_samples=cfg.max_attack_samples,
        )
        records.append({"model": name, "checkpoint": ckpt_path, **metrics, "meta": str(meta)})

    df = pd.DataFrame(records).sort_values(by=["forget_quality", "utility_test_retain"], ascending=False)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"
    md_path = output_dir / "comparison.md"
    plot_path = output_dir / "utility_vs_forget.png"

    df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Utility vs Forget Quality\n\n")
        handle.write(df[[
            "model",
            "utility_test_retain",
            "utility_test_all",
            "forget_train_acc",
            "forget_quality",
            "mia_auc_confidence",
            "mia_auc_delta",
        ]].to_markdown(index=False))
        handle.write("\n")

    _plot_tradeoff(df, str(plot_path))

    best_row = df.iloc[0].to_dict() if not df.empty else {}
    return {
        "csv_path": str(csv_path),
        "markdown_path": str(md_path),
        "plot_path": str(plot_path),
        "best_model": str(best_row.get("model", "")),
    }
