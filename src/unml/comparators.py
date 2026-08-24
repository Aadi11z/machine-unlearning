from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

COMPARATOR_SCHEMA = "unml-comparator-v1"


@dataclass(frozen=True)
class ComparatorSpec:
    comparator_id: str
    family: str
    trainable_parameters: str
    unlearning_compatible: bool
    hypothesis: str


COMPARATOR_SPECS: tuple[ComparatorSpec, ...] = (
    ComparatorSpec(
        "cifar100_zeroshot_clip_v1",
        "zero_shot",
        "none",
        False,
        "How much CIFAR-100 recognition exists before adaptation?",
    ),
    ComparatorSpec(
        "cifar100_linear_probe_v1",
        "linear_probe",
        "classifier_head",
        False,
        "How well do frozen CLIP features support a supervised head?",
    ),
    ComparatorSpec(
        "cifar100_full_ft_v1",
        "full_finetune",
        "all_model_parameters",
        False,
        "What is the supervised adaptation upper reference?",
    ),
    ComparatorSpec(
        "cifar100_lpft_v1",
        "lp_ft",
        "head_then_backbone",
        False,
        "Does staged head-to-backbone optimization improve adaptation?",
    ),
    ComparatorSpec(
        "cifar100_wise_ft_v1",
        "wise_ft",
        "full_finetune_interpolation",
        False,
        "Does interpolation improve ID/shift robustness?",
    ),
    ComparatorSpec(
        "cifar100_lora_delta_interp_v1",
        "lora_delta_interpolation",
        "lora_delta",
        True,
        "Can LoRA adaptation strength trade off utility and forgetting?",
    ),
)


class LinearProbe(nn.Module):
    """A standalone classifier head over frozen CLIP image features."""

    def __init__(self, feature_dim: int, class_count: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(feature_dim, class_count)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def fit_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    steps: int = 500,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> tuple[LinearProbe, dict[str, float]]:
    if train_features.ndim != 2 or val_features.ndim != 2:
        raise ValueError("Linear-probe features must be rank-2 tensors")
    if train_features.shape[1] != val_features.shape[1]:
        raise ValueError("Train and validation feature dimensions differ")
    if steps < 1:
        raise ValueError("Linear-probe steps must be positive")
    class_count = int(torch.max(train_labels).item()) + 1
    probe = LinearProbe(int(train_features.shape[1]), class_count)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=lr, weight_decay=weight_decay
    )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(probe(train_features), train_labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        val_logits = probe(val_features)
        val_loss = nn.functional.cross_entropy(val_logits, val_labels)
        val_accuracy = (val_logits.argmax(dim=1) == val_labels).float().mean()
    return probe, {
        "validation_loss": float(val_loss.item()),
        "validation_accuracy": float(val_accuracy.item()),
    }



def comparator_spec(comparator_id: str) -> ComparatorSpec:
    for spec in COMPARATOR_SPECS:
        if spec.comparator_id == comparator_id:
            return spec
    raise KeyError(f"Unknown comparator id: {comparator_id!r}")


def comparator_manifest_metadata(spec: ComparatorSpec) -> dict[str, Any]:
    return {
        "schema": COMPARATOR_SCHEMA,
        "comparator_id": spec.comparator_id,
        "family": spec.family,
        "trainable_parameters": spec.trainable_parameters,
        "unlearning_compatible": spec.unlearning_compatible,
        "hypothesis": spec.hypothesis,
    }


def validate_comparison(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    """Reject comparisons with incompatible data/model contracts."""
    for field in ("dataset", "prompt_contract", "split"):
        if baseline.get(field) != candidate.get(field):
            raise ValueError(f"Comparison contract mismatch in {field!r}")
    baseline_spec = comparator_spec(str(baseline["comparator_id"]))
    candidate_spec = comparator_spec(str(candidate["comparator_id"]))
    if baseline_spec.family == "zero_shot" or candidate_spec.family == "zero_shot":
        return
    if baseline_spec.unlearning_compatible != candidate_spec.unlearning_compatible:
        raise ValueError("Comparison mixes incompatible unlearning capabilities")


def interpolate_lora_delta(
    base_state: Mapping[str, torch.Tensor],
    adapted_state: Mapping[str, torch.Tensor],
    weight: float,
) -> dict[str, torch.Tensor]:
    """Interpolate an adapter delta without changing the frozen backbone."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("Interpolation weight must be in [0, 1]")
    if set(base_state) != set(adapted_state):
        raise ValueError("Base and adapted adapter keys differ")
    result: dict[str, torch.Tensor] = {}
    for key in base_state:
        base = base_state[key]
        adapted = adapted_state[key]
        if base.shape != adapted.shape or base.dtype != adapted.dtype:
            raise ValueError(f"Adapter tensor contract differs for {key!r}")
        result[key] = base + weight * (adapted - base)
    return result
