from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


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
