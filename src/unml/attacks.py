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


H_TGSD_METHODS = {"h_tgsd", "h_tgsd_no_sibling_preservation"}


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


@dataclass(frozen=True)
class EvaluationBasis:
    target: np.ndarray
    shared: np.ndarray
    request_type: str
    source_models: tuple[str, ...]


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


def _mia_resistance(auc: float) -> float:
    return max(0.0, 1.0 - abs(auc - 0.5) * 2)


def _mia_metrics(
    member_current: np.ndarray,
    nonmember_current: np.ndarray,
    member_base: np.ndarray,
    nonmember_base: np.ndarray,
    *,
    member_loss_current: np.ndarray | None = None,
    nonmember_loss_current: np.ndarray | None = None,
    member_loss_base: np.ndarray | None = None,
    nonmember_loss_base: np.ndarray | None = None,
) -> Dict[str, float]:
    # Membership Inference Attack checks, given a sample and model's output score, whether that sample was in the model's training set(member) or not(non-member)
    # It is a privacy leak, If members have higher confidence than non-members
    # Basically answers, "Can confidence-based attacks tell which sample was in training?"
    sample_count = min(
        len(member_current),
        len(nonmember_current),
        len(member_base),
        len(nonmember_base),
    )
    member_current = member_current[:sample_count]
    nonmember_current = nonmember_current[:sample_count]
    member_base = member_base[:sample_count]
    nonmember_base = nonmember_base[:sample_count]

    y = np.concatenate([np.ones(len(member_current)), np.zeros(len(nonmember_current))])
    conf_scores = np.concatenate([member_current, nonmember_current])
    delta_scores = np.concatenate([member_current - member_base, nonmember_current - nonmember_base])

    auc_conf = _safe_auc(y, conf_scores) # checks if raw true-class confidence leaks membership info
    auc_delta = _safe_auc(y, delta_scores) # checks if change relative to base model leaks membership

    mia_resistance_conf = _mia_resistance(auc_conf)
    mia_resistance_delta = _mia_resistance(auc_delta)
    # If AUC = 0.5 -> resistance = 1.0 (best privacy)
    # if AUC = 0 or 1 -> resistance = 0.0 (worst privacy)
    
    metrics = {
        "mia_auc_confidence": auc_conf,
        "mia_auc_delta": auc_delta,
        "mia_resistance_confidence": mia_resistance_conf,
        "mia_resistance_delta": mia_resistance_delta,
    }
    loss_inputs = (
        member_loss_current,
        nonmember_loss_current,
        member_loss_base,
        nonmember_loss_base,
    )
    if any(value is not None for value in loss_inputs):
        if any(value is None for value in loss_inputs):
            raise ValueError(
                "Loss MIA requires current/base member and nonmember losses"
            )
        loss_lengths = {len(value) for value in loss_inputs if value is not None}
        if len(loss_lengths) != 1 or loss_lengths != {sample_count}:
            raise ValueError(
                "Loss MIA arrays must have equal lengths matching confidence "
                f"inputs ({sample_count}): {loss_lengths}"
            )
        member_loss_current = np.asarray(member_loss_current)
        nonmember_loss_current = np.asarray(nonmember_loss_current)
        member_loss_base = np.asarray(member_loss_base)
        nonmember_loss_base = np.asarray(nonmember_loss_base)
        loss_scores = np.concatenate(
            [-member_loss_current, -nonmember_loss_current]
        )
        loss_delta_scores = np.concatenate(
            [
                member_loss_base - member_loss_current,
                nonmember_loss_base - nonmember_loss_current,
            ]
        )
        auc_loss = _safe_auc(y, loss_scores)
        auc_loss_delta = _safe_auc(y, loss_delta_scores)
        metrics.update(
            {
                "mia_auc_loss": auc_loss,
                "mia_auc_loss_delta": auc_loss_delta,
                "mia_resistance_loss": _mia_resistance(auc_loss),
                "mia_resistance_loss_delta": _mia_resistance(
                    auc_loss_delta
                ),
            }
        )
    return metrics


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


def _load_evaluation_basis(
    names: Sequence[str],
    checkpoints: Sequence[str],
    split: Dict[str, Any],
) -> EvaluationBasis | None:
    candidates = []
    for name, checkpoint in zip(names, checkpoints):
        if name not in H_TGSD_METHODS:
            continue
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        metadata = payload.get("extra", {}).get("h_tgsd_basis")
        if not isinstance(metadata, dict):
            raise ValueError(
                f"{name} checkpoint does not contain H-TGSD basis metadata"
            )
        target = np.asarray(metadata["target"], dtype=np.float64)
        shared = np.asarray(metadata["shared"], dtype=np.float64)
        if target.ndim != 2 or target.shape[1] == 0:
            raise ValueError("H-TGSD target basis must be a non-empty matrix")
        if shared.ndim != 2 or shared.shape[0] != target.shape[0]:
            raise ValueError("H-TGSD shared basis dimension mismatch")
        request_type = str(metadata.get("request_type", ""))
        if request_type != str(split.get("request_type", "")):
            raise ValueError(
                f"H-TGSD basis request_type={request_type!r} does not match "
                f"split request_type={split.get('request_type')!r}"
            )
        if list(metadata.get("target_classes", [])) != list(
            split.get("target_classes", split.get("forget_classes", []))
        ):
            raise ValueError("H-TGSD basis target classes do not match split")
        if list(metadata.get("sibling_classes", [])) != list(
            split.get("sibling_classes", [])
        ):
            raise ValueError("H-TGSD basis sibling classes do not match split")
        candidates.append((name, target, shared, request_type))
    if not candidates:
        return None
    reference = candidates[0]
    for name, target, shared, request_type in candidates[1:]:
        if (
            request_type != reference[3]
            or target.shape != reference[1].shape
            or shared.shape != reference[2].shape
            or not np.allclose(target, reference[1], atol=1e-6, rtol=1e-5)
            or not np.allclose(shared, reference[2], atol=1e-6, rtol=1e-5)
        ):
            raise ValueError(
                f"H-TGSD basis mismatch between {reference[0]} and {name}"
            )
    return EvaluationBasis(
        target=reference[1],
        shared=reference[2],
        request_type=reference[3],
        source_models=tuple(candidate[0] for candidate in candidates),
    )


def _subspace_energy_metrics(
    outputs: Dict[str, np.ndarray | float],
    basis: EvaluationBasis,
    class_groups: Dict[str, list[int]],
) -> tuple[Dict[str, float | int | None], pd.DataFrame]:
    embeddings = np.asarray(outputs.get("embeddings"), dtype=np.float64)
    labels = np.asarray(outputs.get("labels"), dtype=np.int64)
    indices = np.asarray(outputs.get("indices"), dtype=np.int64)
    if embeddings.ndim != 2:
        raise ValueError("Subspace evaluation requires embedding outputs")
    if embeddings.shape[1] != basis.target.shape[0]:
        raise ValueError("Embedding and semantic-basis dimensions differ")
    if labels.shape[0] != embeddings.shape[0] or indices.shape[0] != labels.shape[0]:
        raise ValueError("Subspace evaluation output lengths differ")

    target_energy = np.square(embeddings @ basis.target).sum(axis=1)
    shared_energy = (
        np.square(embeddings @ basis.shared).sum(axis=1)
        if basis.shared.shape[1]
        else np.zeros(embeddings.shape[0], dtype=np.float64)
    )
    hierarchy_group = np.full(labels.shape, "unassigned", dtype=object)
    for group_name in ("target", "sibling", "unrelated"):
        hierarchy_group[np.isin(labels, class_groups[group_name])] = group_name
    if np.any(hierarchy_group == "unassigned"):
        raise ValueError("Some labels are not assigned to a hierarchy group")

    per_sample = pd.DataFrame(
        {
            "index": indices,
            "label": labels,
            "hierarchy_group": hierarchy_group,
            "target_subspace_energy": target_energy,
            "shared_subspace_energy": shared_energy,
        }
    ).sort_values("index")
    measurement_groups = {
        **class_groups,
        "related": sorted(
            set(class_groups["target"]) | set(class_groups["sibling"])
        ),
    }
    metrics: Dict[str, float | int | None] = {
        "target_basis_rank": int(basis.target.shape[1]),
        "shared_basis_rank": int(basis.shared.shape[1]),
    }
    for group_name in (
        "target",
        "sibling",
        "related",
        "unrelated",
        "retain",
        "all",
    ):
        mask = np.isin(labels, measurement_groups[group_name])
        metrics[f"subspace_{group_name}_n"] = int(mask.sum())
        metrics[f"target_subspace_energy_{group_name}"] = (
            float(target_energy[mask].mean()) if mask.any() else None
        )
        metrics[f"shared_subspace_energy_{group_name}"] = (
            float(shared_energy[mask].mean())
            if mask.any() and basis.shared.shape[1]
            else None
        )
    return metrics, per_sample


def _reference_relative_subspace_metrics(
    candidate: Dict[str, float | int | None],
    reference: Dict[str, float | int | None],
    reference_name: str,
) -> Dict[str, float | None]:
    metrics: Dict[str, float | None] = {}
    for energy_name in ("target_subspace_energy", "shared_subspace_energy"):
        for group_name in (
            "target",
            "sibling",
            "related",
            "unrelated",
            "retain",
            "all",
        ):
            key = f"{energy_name}_{group_name}"
            current = candidate.get(key)
            baseline = reference.get(key)
            prefix = f"{key}_vs_{reference_name}"
            if current is None or baseline is None:
                metrics[f"{prefix}_delta"] = None
                metrics[f"{prefix}_ratio"] = None
            else:
                current_value = float(current)
                baseline_value = float(baseline)
                metrics[f"{prefix}_delta"] = current_value - baseline_value
                metrics[f"{prefix}_ratio"] = (
                    current_value / baseline_value
                    if abs(baseline_value) > 1e-12
                    else None
                )
    return metrics


def _run_metadata_columns(
    metadata: Dict[str, Any],
    checkpoint_path: str,
    *,
    dataset_name: str,
    request_name: str | None,
) -> Dict[str, Any]:
    provenance = metadata.get("provenance", {})
    runtime = metadata.get("runtime", metadata.get("benchmark", {}))
    config = metadata.get(
        "unlearning_config", metadata.get("training_config", {})
    )
    architecture = metadata.get("architecture", {})
    checkpoint = Path(checkpoint_path)
    return {
        "dataset": dataset_name,
        "request": request_name,
        "seed": config.get("seed"),
        "git_commit": provenance.get("git_commit"),
        "gpu_name": provenance.get("gpu_name"),
        "cuda_version": provenance.get("cuda_version"),
        "runtime_seconds": runtime.get(
            "total_seconds", runtime.get("training_seconds")
        ),
        "peak_gpu_memory_mb": runtime.get("peak_gpu_memory_mb"),
        "trainable_parameter_count": architecture.get(
            "trainable_parameter_count"
        ),
        "checkpoint_size_bytes": (
            checkpoint.stat().st_size if checkpoint.is_file() else None
        ),
    }


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


def _aligned_value_pairs(
    current: Dict[str, np.ndarray | float],
    reference: Dict[str, np.ndarray | float],
    class_ids: Sequence[int],
    max_samples: int,
    value_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected = set(int(class_id) for class_id in class_ids)
    if value_key not in current or value_key not in reference:
        raise ValueError(f"Missing aligned output field: {value_key}")
    reference_values_by_index = {
        int(index): float(value)
        for index, label, value in zip(
            np.asarray(reference["indices"], dtype=np.int64),
            np.asarray(reference["labels"], dtype=np.int64),
            np.asarray(reference[value_key], dtype=np.float32),
        )
        if int(label) in selected
    }
    current_values = []
    reference_values = []
    for index, label, value in zip(
        np.asarray(current["indices"], dtype=np.int64),
        np.asarray(current["labels"], dtype=np.int64),
        np.asarray(current[value_key], dtype=np.float32),
    ):
        if (
            int(label) not in selected
            or int(index) not in reference_values_by_index
        ):
            continue
        current_values.append(float(value))
        reference_values.append(reference_values_by_index[int(index)])
        if len(current_values) >= max_samples:
            break
    return (
        np.asarray(current_values, dtype=np.float32),
        np.asarray(reference_values, dtype=np.float32),
    )


def _aligned_score_pairs(
    current: Dict[str, np.ndarray | float],
    reference: Dict[str, np.ndarray | float],
    class_ids: Sequence[int],
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _aligned_value_pairs(
        current,
        reference,
        class_ids,
        max_samples,
        "true_scores",
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
    member_loss, member_loss_base = _aligned_value_pairs(
        forget_outputs,
        base_outputs["forget_train"],
        class_groups["target"],
        max_attack_samples,
        "sample_losses",
    )
    nonmember_loss, nonmember_loss_base = _aligned_value_pairs(
        test_outputs,
        base_outputs["test_all"],
        class_groups["target"],
        max_attack_samples,
        "sample_losses",
    )
    attack_count = min(len(member), len(nonmember))

    attacks = _mia_metrics(
        member_current=member[:attack_count],
        nonmember_current=nonmember[:attack_count],
        member_base=member_base[:attack_count],
        nonmember_base=nonmember_base[:attack_count],
        member_loss_current=member_loss[:attack_count],
        nonmember_loss_current=nonmember_loss[:attack_count],
        member_loss_base=member_loss_base[:attack_count],
        nonmember_loss_base=nonmember_loss_base[:attack_count],
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


def _write_subspace_artifacts(
    output_dir: Path,
    model_name: str,
    basis: EvaluationBasis,
    metrics: Dict[str, float | int | None],
    per_sample: pd.DataFrame,
) -> Dict[str, str]:
    artifact_name = _safe_artifact_name(model_name)
    subspace_dir = output_dir / "subspace"
    subspace_dir.mkdir(parents=True, exist_ok=True)
    summary_path = subspace_dir / f"{artifact_name}_summary.json"
    per_sample_path = subspace_dir / f"{artifact_name}_per_sample.csv"
    per_sample.to_csv(per_sample_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": model_name,
                "request_type": basis.request_type,
                "basis_source_models": list(basis.source_models),
                "target_basis_rank": int(basis.target.shape[1]),
                "shared_basis_rank": int(basis.shared.shape[1]),
                "energy_definition": "squared L2 norm of basis coordinates",
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


def _pareto_mask_minimize_x_maximize_y(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> np.ndarray:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Pareto inputs must be equal-length vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Pareto inputs must contain only finite values")
    optimal = np.ones(x.shape[0], dtype=bool)
    for index in range(x.shape[0]):
        dominated = (
            (x <= x[index])
            & (y >= y[index])
            & ((x < x[index]) | (y > y[index]))
        )
        optimal[index] = not bool(dominated.any())
    return optimal


def _behavioral_tradeoff_data(
    df: pd.DataFrame,
    class_groups: Dict[str, list[int]],
) -> tuple[pd.DataFrame, str, str]:
    preservation_metric = (
        "sibling_test_macro_acc"
        if class_groups["sibling"]
        else "unrelated_test_macro_acc"
    )
    required = {"model", "target_test_acc", preservation_metric}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Behavioral tradeoff requires comparison columns: {missing}"
        )
    plot_data = df[
        ["model", "target_test_acc", preservation_metric]
    ].copy()
    plot_data = plot_data.dropna(
        subset=["target_test_acc", preservation_metric]
    )
    if plot_data.empty:
        raise ValueError("Behavioral tradeoff has no complete model rows")
    plot_data["behavior_pareto_optimal"] = (
        _pareto_mask_minimize_x_maximize_y(
            plot_data["target_test_acc"],
            plot_data[preservation_metric],
        )
    )
    preservation_label = (
        "Sibling Macro Accuracy"
        if class_groups["sibling"]
        else "Unrelated-Retain Macro Accuracy"
    )
    return plot_data, preservation_metric, preservation_label


def _plot_behavioral_tradeoff(
    df: pd.DataFrame,
    class_groups: Dict[str, list[int]],
    out_path: Path,
) -> Dict[str, Any]:
    plot_data, preservation_metric, preservation_label = (
        _behavioral_tradeoff_data(df, class_groups)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = np.where(
        plot_data["behavior_pareto_optimal"], "#c43c39", "#4c78a8"
    )
    ax.scatter(
        plot_data["target_test_acc"],
        plot_data[preservation_metric],
        c=colors,
        s=85,
        edgecolors="white",
        linewidths=0.7,
    )
    for _, row in plot_data.iterrows():
        ax.annotate(
            str(row["model"]),
            (row["target_test_acc"], row[preservation_metric]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Target Test Accuracy (lower is better)")
    ax.set_ylabel(f"{preservation_label} (higher is better)")
    ax.set_title("Behavioral Forgetting-Retention Tradeoff")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return {
        "path": str(out_path),
        "target_metric": "target_test_acc",
        "preservation_metric": preservation_metric,
        "pareto_models": plot_data.loc[
            plot_data["behavior_pareto_optimal"], "model"
        ].astype(str).tolist(),
    }


def _plot_subspace_summary(
    df: pd.DataFrame,
    basis: EvaluationBasis,
    out_path: Path,
) -> Dict[str, Any]:
    target_metric = (
        "target_subspace_energy_target_vs_finetuned_ratio"
    )
    metrics = [target_metric]
    labels = ["Target energy / fine-tuned"]
    shared_metric = (
        "shared_subspace_energy_related_vs_finetuned_ratio"
    )
    if basis.shared.shape[1] and shared_metric in df.columns:
        metrics.append(shared_metric)
        labels.append("Shared related energy / fine-tuned")
    required = {"model", *metrics}
    missing = sorted(required - set(df.columns))
    if missing:
        print(
            f"[warn] Skipping subspace plot because required columns are missing: {missing}",
            flush=True,
        )
        return {
            "path": None,
            "metrics": metrics,
            "basis_source_models": list(basis.source_models),
            "skipped": True,
            "reason": f"missing columns: {missing}",
        }
    plot_data = df[["model", *metrics]].dropna(
        subset=[target_metric]
    )
    if plot_data.empty:
        print(
            "[warn] Skipping subspace plot because no complete model rows are available",
            flush=True,
        )
        return {
            "path": None,
            "metrics": metrics,
            "basis_source_models": list(basis.source_models),
            "skipped": True,
            "reason": "no complete model rows",
        }
    positions = np.arange(len(plot_data))
    width = 0.75 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(7, len(plot_data) * 0.85), 5))
    for metric_index, (metric, label) in enumerate(zip(metrics, labels)):
        offset = (metric_index - (len(metrics) - 1) / 2) * width
        ax.bar(
            positions + offset,
            plot_data[metric],
            width=width,
            label=label,
        )
    ax.axhline(
        1.0,
        color="#555555",
        linestyle="--",
        linewidth=1,
        label="Fine-tuned reference",
    )
    ax.set_xticks(positions, plot_data["model"], rotation=30, ha="right")
    ax.set_ylabel("Energy ratio")
    ax.set_title("Canonical Semantic-Subspace Energy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return {
        "path": str(out_path),
        "metrics": metrics,
        "basis_source_models": list(basis.source_models),
    }


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
    evaluation_basis = _load_evaluation_basis(
        cfg.candidate_names, cfg.candidate_checkpoints, split
    )
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
    subspace_artifacts = {}
    subspace_metrics_by_model: Dict[
        str, Dict[str, float | int | None]
    ] = {}
    subspace_samples_by_model: Dict[str, pd.DataFrame] = {}
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
        if evaluation_basis is not None:
            subspace_metrics, subspace_samples = _subspace_energy_metrics(
                details["test_outputs"], evaluation_basis, class_groups
            )
            metrics.update(subspace_metrics)
            subspace_metrics_by_model[name] = subspace_metrics
            subspace_samples_by_model[name] = subspace_samples
        hierarchy_artifacts[name] = _write_hierarchy_artifacts(
            output_dir, name, details
        )
        records.append(
            {
                "model": name,
                "checkpoint": ckpt_path,
                **_run_metadata_columns(
                    meta,
                    ckpt_path,
                    dataset_name=dataset_spec.name,
                    request_name=split.get("request_name"),
                ),
                **metrics,
                "meta": json.dumps(meta, default=str, sort_keys=True),
            }
        )
        update_unlearn_with_attacks(name, metrics)

    if evaluation_basis is not None:
        references = {
            reference_name: subspace_metrics_by_model[reference_name]
            for reference_name in ("finetuned", "retrain_oracle")
            if reference_name in subspace_metrics_by_model
        }
        for record in records:
            name = str(record["model"])
            model_metrics = subspace_metrics_by_model[name]
            relative_metrics: Dict[str, float | None] = {}
            for reference_name, reference_metrics in references.items():
                relative_metrics.update(
                    _reference_relative_subspace_metrics(
                        model_metrics, reference_metrics, reference_name
                    )
                )
            record.update(relative_metrics)
            complete_metrics = {**model_metrics, **relative_metrics}
            subspace_artifacts[name] = _write_subspace_artifacts(
                output_dir,
                name,
                evaluation_basis,
                complete_metrics,
                subspace_samples_by_model[name],
            )

    df = pd.DataFrame(records).sort_values(by=["forget_quality", "utility_test_retain"], ascending=False)

    csv_path = output_dir / "comparison.csv"
    md_path = output_dir / "comparison.md"
    plot_path = output_dir / "utility_vs_forget.png"
    behavioral_plot_path = output_dir / "behavioral_tradeoff.png"
    subspace_plot_path = output_dir / "semantic_subspace.png"

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
            "mia_auc_loss",
            "mia_auc_loss_delta",
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
        if evaluation_basis is not None:
            markdown_columns.extend(
                [
                    "target_subspace_energy_target",
                    "target_subspace_energy_target_vs_finetuned_ratio",
                    "target_subspace_energy_target_vs_retrain_oracle_ratio",
                    "shared_subspace_energy_related",
                    "shared_subspace_energy_related_vs_finetuned_ratio",
                    "shared_subspace_energy_related_vs_retrain_oracle_ratio",
                ]
            )
        markdown_columns = [
            column for column in markdown_columns if column in df.columns
        ]
        try:
            markdown = df[markdown_columns].to_markdown(index=False)
        except ImportError:
            markdown = df[markdown_columns].to_string(index=False)
        handle.write(markdown)
        handle.write("\n")

    _plot_tradeoff(df, str(plot_path))
    behavioral_plot = _plot_behavioral_tradeoff(
        df, class_groups, behavioral_plot_path
    )
    subspace_plot = (
        _plot_subspace_summary(df, evaluation_basis, subspace_plot_path)
        if evaluation_basis is not None
        else None
    )
    plot_manifest_path = output_dir / "plot_manifest.json"
    with plot_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "historical_composite_plot": str(plot_path),
                "behavioral_tradeoff": behavioral_plot,
                "semantic_subspace": subspace_plot,
            },
            handle,
            indent=2,
        )

    best_row = df.iloc[0].to_dict() if not df.empty else {}
    return {
        "csv_path": str(csv_path),
        "markdown_path": str(md_path),
        "plot_path": str(plot_path),
        "behavioral_plot": behavioral_plot,
        "subspace_plot": subspace_plot,
        "plot_manifest_path": str(plot_manifest_path),
        "hierarchy_artifacts": hierarchy_artifacts,
        "oracle_artifacts": oracle_artifacts,
        "subspace_artifacts": subspace_artifacts,
        "best_model": str(best_row.get("model", "")),
    }
