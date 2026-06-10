from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor, CLIPTokenizer

from .data import (
    build_loaders,
    build_text_inputs,
    load_split_metadata,
    validate_checkpoint_dataset,
)
from .evaluate import (
    build_class_text_features,
    collect_classification_outputs,
    summarize_classification_group,
)
from .model import load_checkpoint
from helpers.tracker import update_unlearn_with_attacks
from .utils import get_device


@dataclass
class AttackConfig:
    data_dir: str
    split_path: str
    dataset_name: str
    model_name: str
    base_checkpoint: str
    candidate_checkpoints: Sequence[str]
    candidate_names: Sequence[str]
    output_dir: str
    prompt_template: str = "a photo of a {}"
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2
    non_blocking: bool = False
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
    # AUC = 0.5: no separability (random guessing).
    # AUC > 0.5: attacker can distinguish members from non-members.
    # AUC = 1.0: perfect separation.
    # AUC < 0.5: reversed direction (if attacker flips the decision rule, it effectively becomes > 0.5).
    # AUC is the probability that a random member gets a higher score than a random non-member.


def _mia_metrics(member_current: np.ndarray, nonmember_current: np.ndarray, member_base: np.ndarray, nonmember_base: np.ndarray) -> Dict[str, float]:
    # Membership Inference Attack checks, given a sample and model's output score, whether that sample was in the model's training set(member) or not(non-member)
    # It is a privacy leak, If members have higher confidence than non-members
    # Basically answers, "Can confidence-based attacks tell which sample was in training?"
    member_current, nonmember_current = _truncate_equal(member_current, nonmember_current)
    member_base, nonmember_base = _truncate_equal(member_base, nonmember_base)

    y = np.concatenate([np.ones(len(member_current)), np.zeros(len(nonmember_current))])
    conf_scores = np.concatenate([member_current, nonmember_current])
    delta_scores = np.concatenate([member_current - member_base, nonmember_current - nonmember_base])

    auc_conf = _safe_auc(y, conf_scores) # checks if raw true-class confidence leaks membership info
    auc_delta = _safe_auc(y, delta_scores) # checks if change relative to base model leaks membership

    mia_resistance_conf = max(0.0, 1.0 - abs(auc_conf - 0.5) * 2)
    mia_resistance_delta = max(0.0, 1.0 - abs(auc_delta - 0.5) * 2)
    # If AUC = 0.5 -> resistance = 1.0 (best privacy)
    # if AUC = 0 or 1 -> resistance = 0.0 (worst privacy)
    
    return {
        "mia_auc_confidence": auc_conf,
        "mia_auc_delta": auc_delta,
        "mia_resistance_confidence": mia_resistance_conf,
        "mia_resistance_delta": mia_resistance_delta,
    }


def _group_classes(
    split: Dict[str, Any], num_classes: int
) -> Dict[str, list[int]]:
    target = sorted(
        set(split.get("target_classes", split.get("forget_classes", [])))
    )
    sibling = sorted(set(split.get("sibling_classes", [])))
    unrelated = sorted(set(split.get("unrelated_classes", [])))
    if not unrelated:
        unrelated = sorted(set(range(num_classes)) - set(target) - set(sibling))
    if not target:
        raise ValueError("Evaluation requires at least one target class")
    named_groups = {
        "target": set(target),
        "sibling": set(sibling),
        "unrelated": set(unrelated),
    }
    for group_name, class_ids in named_groups.items():
        invalid = sorted(
            class_id
            for class_id in class_ids
            if class_id < 0 or class_id >= num_classes
        )
        if invalid:
            raise ValueError(
                f"{group_name} classes outside [0, {num_classes - 1}]: "
                f"{invalid}"
            )
    for left, right in (
        ("target", "sibling"),
        ("target", "unrelated"),
        ("sibling", "unrelated"),
    ):
        overlap = sorted(named_groups[left] & named_groups[right])
        if overlap:
            raise ValueError(
                f"Evaluation class groups {left}/{right} overlap: {overlap}"
            )
    missing = sorted(set(range(num_classes)) - set().union(*named_groups.values()))
    if missing:
        raise ValueError(
            f"Evaluation class groups do not cover classes: {missing}"
        )
    return {
        "target": target,
        "sibling": sibling,
        "unrelated": unrelated,
        "retain": sorted(set(sibling) | set(unrelated)),
        "all": list(range(num_classes)),
    }


def _validate_candidates(
    names: Sequence[str], checkpoints: Sequence[str]
) -> None:
    if len(names) != len(checkpoints):
        raise ValueError(
            "candidate_names and candidate_checkpoints must have equal length"
        )
    if not names:
        raise ValueError("At least one candidate checkpoint is required")
    duplicates = sorted(
        {name for name in names if list(names).count(name) > 1}
    )
    if duplicates:
        raise ValueError(f"Candidate names must be unique: {duplicates}")


def _reference_outputs(
    model,
    loaders: Dict[str, DataLoader],
    class_text_inputs: Dict[str, torch.Tensor],
    device: torch.device,
    non_blocking: bool,
) -> Dict[str, Dict[str, np.ndarray | float]]:
    text_features = build_class_text_features(model, class_text_inputs, device)
    common = {
        "class_text_inputs": class_text_inputs,
        "device": device,
        "class_text_features": text_features,
        "non_blocking": non_blocking,
    }
    return {
        "test_all": collect_classification_outputs(
            model, loaders["test_all"], **common
        ),
        "forget_train": collect_classification_outputs(
            model, loaders["forget"], **common
        ),
    }


def _aligned_score_pairs(
    current: Dict[str, np.ndarray | float],
    reference: Dict[str, np.ndarray | float],
    class_ids: Sequence[int],
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = set(int(class_id) for class_id in class_ids)
    reference_scores = {
        int(index): float(score)
        for index, label, score in zip(
            np.asarray(reference["indices"], dtype=np.int64),
            np.asarray(reference["labels"], dtype=np.int64),
            np.asarray(reference["true_scores"], dtype=np.float32),
        )
        if int(label) in selected
    }
    current_values = []
    reference_values = []
    for index, label, score in zip(
        np.asarray(current["indices"], dtype=np.int64),
        np.asarray(current["labels"], dtype=np.int64),
        np.asarray(current["true_scores"], dtype=np.float32),
    ):
        if int(label) not in selected or int(index) not in reference_scores:
            continue
        current_values.append(float(score))
        reference_values.append(reference_scores[int(index)])
        if len(current_values) >= max_samples:
            break
    return (
        np.asarray(current_values, dtype=np.float32),
        np.asarray(reference_values, dtype=np.float32),
    )


def _oracle_relative_distances(
    candidate: Dict[str, np.ndarray | float],
    oracle: Dict[str, np.ndarray | float],
    class_groups: Dict[str, list[int]],
) -> tuple[Dict[str, float | int | None], pd.DataFrame]:
    required = ("indices", "labels", "probabilities", "embeddings")
    for name, outputs in (("candidate", candidate), ("oracle", oracle)):
        missing = [key for key in required if key not in outputs]
        if missing:
            raise ValueError(
                f"{name} outputs missing oracle-comparison fields: {missing}"
            )

    oracle_indices = np.asarray(oracle["indices"], dtype=np.int64)
    if np.unique(oracle_indices).size != oracle_indices.size:
        raise ValueError("Oracle outputs contain duplicate dataset indices")
    oracle_rows = {
        int(index): row for row, index in enumerate(oracle_indices)
    }

    candidate_indices = np.asarray(candidate["indices"], dtype=np.int64)
    if np.unique(candidate_indices).size != candidate_indices.size:
        raise ValueError("Candidate outputs contain duplicate dataset indices")
    if set(candidate_indices.tolist()) != set(oracle_indices.tolist()):
        raise ValueError(
            "Candidate and oracle outputs do not cover identical dataset indices"
        )

    candidate_labels = np.asarray(candidate["labels"], dtype=np.int64)
    oracle_labels = np.asarray(oracle["labels"], dtype=np.int64)
    candidate_probabilities = np.asarray(
        candidate["probabilities"], dtype=np.float64
    )
    oracle_probabilities = np.asarray(
        oracle["probabilities"], dtype=np.float64
    )
    candidate_embeddings = np.asarray(
        candidate["embeddings"], dtype=np.float64
    )
    oracle_embeddings = np.asarray(oracle["embeddings"], dtype=np.float64)

    aligned_oracle_rows = np.asarray(
        [oracle_rows[int(index)] for index in candidate_indices],
        dtype=np.int64,
    )
    aligned_oracle_labels = oracle_labels[aligned_oracle_rows]
    if not np.array_equal(candidate_labels, aligned_oracle_labels):
        raise ValueError(
            "Candidate and oracle labels disagree after dataset-index alignment"
        )
    aligned_oracle_probabilities = oracle_probabilities[aligned_oracle_rows]
    aligned_oracle_embeddings = oracle_embeddings[aligned_oracle_rows]
    if candidate_probabilities.shape != aligned_oracle_probabilities.shape:
        raise ValueError("Candidate and oracle probability shapes differ")
    if candidate_embeddings.shape != aligned_oracle_embeddings.shape:
        raise ValueError("Candidate and oracle embedding shapes differ")

    epsilon = np.finfo(np.float64).eps
    candidate_probabilities = np.clip(
        candidate_probabilities, epsilon, None
    )
    aligned_oracle_probabilities = np.clip(
        aligned_oracle_probabilities, epsilon, None
    )
    candidate_probabilities /= candidate_probabilities.sum(
        axis=1, keepdims=True
    )
    aligned_oracle_probabilities /= aligned_oracle_probabilities.sum(
        axis=1, keepdims=True
    )
    prediction_kl = np.sum(
        aligned_oracle_probabilities
        * (
            np.log(aligned_oracle_probabilities)
            - np.log(candidate_probabilities)
        ),
        axis=1,
    )

    candidate_norms = np.linalg.norm(
        candidate_embeddings, axis=1, keepdims=True
    )
    oracle_norms = np.linalg.norm(
        aligned_oracle_embeddings, axis=1, keepdims=True
    )
    if np.any(candidate_norms == 0) or np.any(oracle_norms == 0):
        raise ValueError("Oracle comparison received a zero-norm embedding")
    embedding_cosine = 1.0 - np.sum(
        (candidate_embeddings / candidate_norms)
        * (aligned_oracle_embeddings / oracle_norms),
        axis=1,
    )
    embedding_cosine = np.maximum(embedding_cosine, 0.0)

    hierarchy_group = np.full(candidate_labels.shape, "unassigned", dtype=object)
    for group_name in ("target", "sibling", "unrelated"):
        hierarchy_group[
            np.isin(candidate_labels, class_groups[group_name])
        ] = group_name
    if np.any(hierarchy_group == "unassigned"):
        raise ValueError("Some labels are not assigned to a hierarchy group")

    per_sample = pd.DataFrame(
        {
            "index": candidate_indices,
            "label": candidate_labels,
            "hierarchy_group": hierarchy_group,
            "is_retain": np.isin(
                candidate_labels, class_groups["retain"]
            ),
            "prediction_kl_oracle_to_candidate": prediction_kl,
            "embedding_cosine_distance_from_oracle": embedding_cosine,
        }
    ).sort_values("index")

    metrics: Dict[str, float | int | None] = {}
    for group_name in ("target", "sibling", "unrelated", "retain", "all"):
        mask = np.isin(candidate_labels, class_groups[group_name])
        metrics[f"oracle_distance_{group_name}_n"] = int(mask.sum())
        metrics[f"oracle_prediction_kl_{group_name}"] = (
            float(prediction_kl[mask].mean()) if mask.any() else None
        )
        metrics[f"oracle_embedding_cosine_{group_name}"] = (
            float(embedding_cosine[mask].mean()) if mask.any() else None
        )
    return metrics, per_sample


def _evaluate_model(
    model,
    loaders: Dict[str, DataLoader],
    class_text_inputs: Dict[str, torch.Tensor],
    device: torch.device,
    max_attack_samples: int,
    class_groups: Dict[str, list[int]],
    class_names: Sequence[str],
    base_outputs: Dict[str, Dict[str, np.ndarray | float]],
    non_blocking: bool = False,
) -> tuple[Dict[str, float | int | None], Dict[str, Any]]:
    model_text_features = build_class_text_features(
        model, class_text_inputs, device
    )
    common = {
        "class_text_inputs": class_text_inputs,
        "device": device,
        "class_text_features": model_text_features,
        "non_blocking": non_blocking,
    }
    test_outputs = collect_classification_outputs(
        model,
        loaders["test_all"],
        include_probabilities=True,
        include_embeddings=True,
        **common,
    )
    forget_outputs = collect_classification_outputs(
        model,
        loaders["forget"],
        **common,
    )
    num_classes = len(class_names)
    groups = {
        group_name: summarize_classification_group(
            test_outputs, class_ids, num_classes
        )
        for group_name, class_ids in class_groups.items()
    }
    forget_train = summarize_classification_group(
        forget_outputs, class_groups["target"], num_classes
    )
    member, member_base = _aligned_score_pairs(
        forget_outputs,
        base_outputs["forget_train"],
        class_groups["target"],
        max_attack_samples,
    )
    nonmember, nonmember_base = _aligned_score_pairs(
        test_outputs,
        base_outputs["test_all"],
        class_groups["target"],
        max_attack_samples,
    )
    attack_count = min(len(member), len(nonmember))

    attacks = _mia_metrics(
        member_current=member[:attack_count],
        nonmember_current=nonmember[:attack_count],
        member_base=member_base[:attack_count],
        nonmember_base=nonmember_base[:attack_count],
    )

    forget_train_accuracy = float(forget_train["accuracy"] or 0.0)
    forget_drop = max(0.0, 1.0 - forget_train_accuracy)
    forget_quality = (forget_drop + attacks["mia_resistance_confidence"] + attacks["mia_resistance_delta"]) / 3.0

    scalar_metrics = {
        "utility_test_all": groups["all"]["accuracy"],
        "utility_test_retain": groups["retain"]["accuracy"],
        "target_test_acc": groups["target"]["accuracy"],
        "target_test_macro_acc": groups["target"]["macro_accuracy"],
        "target_test_n": groups["target"]["n"],
        "sibling_test_acc": groups["sibling"]["accuracy"],
        "sibling_test_macro_acc": groups["sibling"]["macro_accuracy"],
        "sibling_test_n": groups["sibling"]["n"],
        "unrelated_test_acc": groups["unrelated"]["accuracy"],
        "unrelated_test_macro_acc": groups["unrelated"]["macro_accuracy"],
        "unrelated_test_n": groups["unrelated"]["n"],
        "forget_train_acc": forget_train_accuracy,
        "forget_drop": forget_drop,
        "forget_quality": forget_quality,
        **attacks,
    }
    details = {
        "class_names": list(class_names),
        "class_groups": class_groups,
        "groups": groups,
        "forget_train": forget_train,
        "mia_sample_count": attack_count,
        "test_outputs": test_outputs,
    }
    return scalar_metrics, details


def _safe_artifact_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return normalized or "model"


def _write_hierarchy_artifacts(
    output_dir: Path,
    model_name: str,
    details: Dict[str, Any],
) -> Dict[str, str]:
    artifact_name = _safe_artifact_name(model_name)
    hierarchy_dir = output_dir / "hierarchy"
    hierarchy_dir.mkdir(parents=True, exist_ok=True)
    class_names = details["class_names"]
    all_group = details["groups"]["all"]
    confusion = np.asarray(all_group["confusion_matrix"], dtype=np.int64)

    class_group_lookup = {}
    for group_name in ("target", "sibling", "unrelated"):
        for class_id in details["class_groups"][group_name]:
            class_group_lookup[int(class_id)] = group_name
    per_class_rows = []
    for row in all_group["per_class"]:
        class_id = int(row["class_id"])
        per_class_rows.append(
            {
                **row,
                "class_name": class_names[class_id],
                "group": class_group_lookup.get(class_id, "other"),
            }
        )

    per_class_path = hierarchy_dir / f"{artifact_name}_per_class.csv"
    confusion_path = hierarchy_dir / f"{artifact_name}_confusion.csv"
    group_confusion_path = (
        hierarchy_dir / f"{artifact_name}_group_confusion.csv"
    )
    metrics_path = hierarchy_dir / f"{artifact_name}_groups.json"
    pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False)
    pd.DataFrame(
        confusion,
        index=class_names,
        columns=class_names,
    ).rename_axis("actual").to_csv(confusion_path)
    hierarchy_group_names = ["target", "sibling", "unrelated"]
    group_confusion = np.zeros((3, 3), dtype=np.int64)
    for actual_index, actual_group in enumerate(hierarchy_group_names):
        actual_classes = details["class_groups"][actual_group]
        for predicted_index, predicted_group in enumerate(
            hierarchy_group_names
        ):
            predicted_classes = details["class_groups"][predicted_group]
            if actual_classes and predicted_classes:
                group_confusion[actual_index, predicted_index] = int(
                    confusion[np.ix_(actual_classes, predicted_classes)].sum()
                )
    pd.DataFrame(
        group_confusion,
        index=hierarchy_group_names,
        columns=hierarchy_group_names,
    ).rename_axis("actual_group").to_csv(group_confusion_path)
    serializable_groups = {}
    for group_name, group in details["groups"].items():
        serializable_groups[group_name] = {
            key: (
                value.tolist()
                if isinstance(value, np.ndarray)
                else value
            )
            for key, value in group.items()
        }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": model_name,
                "class_names": class_names,
                "class_groups": details["class_groups"],
                "groups": serializable_groups,
                "mia_sample_count": details["mia_sample_count"],
            },
            handle,
            indent=2,
        )
    return {
        "per_class_path": str(per_class_path),
        "confusion_path": str(confusion_path),
        "group_confusion_path": str(group_confusion_path),
        "group_metrics_path": str(metrics_path),
    }


def _write_oracle_artifacts(
    output_dir: Path,
    model_name: str,
    metrics: Dict[str, float | int | None],
    per_sample: pd.DataFrame,
) -> Dict[str, str]:
    artifact_name = _safe_artifact_name(model_name)
    oracle_dir = output_dir / "oracle_reference"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    summary_path = oracle_dir / f"{artifact_name}_summary.json"
    per_sample_path = oracle_dir / f"{artifact_name}_per_sample.csv"
    per_sample.to_csv(per_sample_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": model_name,
                "reference": "retrain_oracle",
                "prediction_distance": "KL(oracle || candidate)",
                "embedding_distance": "1 - cosine(candidate, oracle)",
                "metrics": metrics,
            },
            handle,
            indent=2,
        )
    return {
        "summary_path": str(summary_path),
        "per_sample_path": str(per_sample_path),
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


def run_attack_comparison(cfg: AttackConfig) -> Dict[str, Any]:
    _validate_candidates(cfg.candidate_names, cfg.candidate_checkpoints)
    device = get_device(cfg.device)

    image_processor = CLIPImageProcessor.from_pretrained(cfg.model_name)
    tokenizer = CLIPTokenizer.from_pretrained(cfg.model_name)
    split, dataset_spec, class_names = load_split_metadata(
        cfg.split_path, cfg.dataset_name
    )
    class_text_inputs = build_text_inputs(
        tokenizer,
        class_names=class_names,
        template=cfg.prompt_template,
    )
    class_text_inputs = {k: v.to(device) for k, v in class_text_inputs.items()}

    loaders = build_loaders(
        data_dir=cfg.data_dir,
        split_path=cfg.split_path,
        image_processor=image_processor,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        dataset_name=dataset_spec.name,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
    )

    base_model, base_meta = load_checkpoint(cfg.base_checkpoint, map_location=device)
    validate_checkpoint_dataset(base_meta, dataset_spec.name, cfg.base_checkpoint)
    base_model = base_model.to(device).eval()
    class_groups = _group_classes(split, len(class_names))
    base_outputs = _reference_outputs(
        base_model,
        loaders,
        class_text_inputs,
        device,
        cfg.non_blocking,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, float | int | str | None]] = []
    hierarchy_artifacts = {}
    oracle_artifacts = {}
    oracle_outputs = None
    oracle_index = (
        list(cfg.candidate_names).index("retrain_oracle")
        if "retrain_oracle" in cfg.candidate_names
        else None
    )
    cached_oracle_evaluation = None
    if oracle_index is not None:
        oracle_checkpoint = cfg.candidate_checkpoints[oracle_index]
        oracle_model, oracle_meta = load_checkpoint(
            oracle_checkpoint, map_location=device
        )
        validate_checkpoint_dataset(
            oracle_meta, dataset_spec.name, oracle_checkpoint
        )
        oracle_model = oracle_model.to(device).eval()
        cached_oracle_evaluation = _evaluate_model(
            model=oracle_model,
            loaders=loaders,
            class_text_inputs=class_text_inputs,
            device=device,
            max_attack_samples=cfg.max_attack_samples,
            class_groups=class_groups,
            class_names=class_names,
            base_outputs=base_outputs,
            non_blocking=cfg.non_blocking,
        )
        oracle_outputs = cached_oracle_evaluation[1]["test_outputs"]
        del oracle_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for name, ckpt_path in zip(cfg.candidate_names, cfg.candidate_checkpoints):
        if name == "retrain_oracle" and cached_oracle_evaluation is not None:
            metrics, details = cached_oracle_evaluation
            meta = oracle_meta
        else:
            model, meta = load_checkpoint(ckpt_path, map_location=device)
            validate_checkpoint_dataset(meta, dataset_spec.name, ckpt_path)
            model = model.to(device).eval()
            metrics, details = _evaluate_model(
                model=model,
                loaders=loaders,
                class_text_inputs=class_text_inputs,
                device=device,
                max_attack_samples=cfg.max_attack_samples,
                class_groups=class_groups,
                class_names=class_names,
                base_outputs=base_outputs,
                non_blocking=cfg.non_blocking,
            )
        if oracle_outputs is not None:
            oracle_metrics, per_sample = _oracle_relative_distances(
                details["test_outputs"], oracle_outputs, class_groups
            )
            metrics.update(oracle_metrics)
            oracle_artifacts[name] = _write_oracle_artifacts(
                output_dir, name, oracle_metrics, per_sample
            )
        hierarchy_artifacts[name] = _write_hierarchy_artifacts(
            output_dir, name, details
        )
        records.append({"model": name, "checkpoint": ckpt_path, **metrics, "meta": str(meta)})
        update_unlearn_with_attacks(name, metrics)

    df = pd.DataFrame(records).sort_values(by=["forget_quality", "utility_test_retain"], ascending=False)

    csv_path = output_dir / "comparison.csv"
    md_path = output_dir / "comparison.md"
    plot_path = output_dir / "utility_vs_forget.png"

    df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Utility vs Forget Quality\n\n")
        markdown_columns = [
            "model",
            "utility_test_retain",
            "utility_test_all",
            "target_test_acc",
            "target_test_macro_acc",
            "sibling_test_acc",
            "sibling_test_macro_acc",
            "unrelated_test_acc",
            "unrelated_test_macro_acc",
            "forget_train_acc",
            "forget_quality",
            "mia_auc_confidence",
            "mia_auc_delta",
        ]
        if oracle_outputs is not None:
            markdown_columns.extend(
                [
                    "oracle_prediction_kl_target",
                    "oracle_prediction_kl_sibling",
                    "oracle_prediction_kl_unrelated",
                    "oracle_prediction_kl_all",
                    "oracle_embedding_cosine_target",
                    "oracle_embedding_cosine_sibling",
                    "oracle_embedding_cosine_unrelated",
                    "oracle_embedding_cosine_all",
                ]
            )
        handle.write(df[markdown_columns].to_markdown(index=False))
        handle.write("\n")

    _plot_tradeoff(df, str(plot_path))

    best_row = df.iloc[0].to_dict() if not df.empty else {}
    return {
        "csv_path": str(csv_path),
        "markdown_path": str(md_path),
        "plot_path": str(plot_path),
        "hierarchy_artifacts": hierarchy_artifacts,
        "oracle_artifacts": oracle_artifacts,
        "best_model": str(best_row.get("model", "")),
    }
