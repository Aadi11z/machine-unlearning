from __future__ import annotations

import pytest

from unml.comparators import (
    COMPARATOR_SPECS,
    comparator_manifest_metadata,
    comparator_spec,
    validate_comparison,
)


def test_comparator_ids_are_unique_and_hypotheses_are_explicit() -> None:
    ids = [spec.comparator_id for spec in COMPARATOR_SPECS]
    assert len(ids) == len(set(ids))
    assert all(spec.hypothesis for spec in COMPARATOR_SPECS)


def test_comparator_metadata_preserves_family_identity() -> None:
    spec = comparator_spec("cifar100_linear_probe_v1")
    metadata = comparator_manifest_metadata(spec)
    assert metadata["family"] == "linear_probe"
    assert metadata["trainable_parameters"] == "classifier_head"


def test_comparison_rejects_mismatched_prompt_contract() -> None:
    common = {"dataset": "cifar100", "split": {"id": "canonical"}}
    with pytest.raises(ValueError, match="prompt_contract"):
        validate_comparison(
            {**common, "prompt_contract": "a", "comparator_id": "cifar100_linear_probe_v1"},
            {**common, "prompt_contract": "b", "comparator_id": "cifar100_full_ft_v1"},
        )


def test_comparison_rejects_unlearning_capability_mismatch() -> None:
    common = {"dataset": "cifar100", "prompt_contract": "p", "split": "s"}
    with pytest.raises(ValueError, match="unlearning capabilities"):
        validate_comparison(
            {**common, "comparator_id": "cifar100_lora_delta_interp_v1"},
            {**common, "comparator_id": "cifar100_linear_probe_v1"},
        )
