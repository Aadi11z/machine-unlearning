from __future__ import annotations

import json

import numpy as np
import pytest

from unml.attacks import (
    _aligned_score_pairs,
    _group_classes,
    _safe_artifact_name,
    _validate_candidates,
    _write_hierarchy_artifacts,
)
from unml.evaluate import summarize_classification_group


def _outputs(
    labels: list[int],
    predictions: list[int],
    indices: list[int] | None = None,
    scores: list[float] | None = None,
) -> dict:
    count = len(labels)
    return {
        "labels": np.asarray(labels, dtype=np.int64),
        "predictions": np.asarray(predictions, dtype=np.int64),
        "indices": np.asarray(
            indices if indices is not None else list(range(count)),
            dtype=np.int64,
        ),
        "true_scores": np.asarray(
            scores if scores is not None else [0.5] * count,
            dtype=np.float32,
        ),
        "loss": 0.0,
    }


def test_group_summary_reports_micro_macro_and_confusion() -> None:
    outputs = _outputs(
        labels=[0, 0, 1, 1, 2],
        predictions=[0, 1, 1, 1, 0],
    )

    summary = summarize_classification_group(
        outputs, class_ids=[0, 1], num_classes=3
    )

    assert summary["n"] == 4
    assert summary["accuracy"] == pytest.approx(0.75)
    assert summary["macro_accuracy"] == pytest.approx(0.75)
    assert summary["confusion_matrix"][0, 0] == 1
    assert summary["confusion_matrix"][0, 1] == 1
    assert summary["confusion_matrix"][1, 1] == 2


def test_empty_sibling_group_is_explicitly_not_applicable() -> None:
    summary = summarize_classification_group(
        _outputs(labels=[0], predictions=[0]),
        class_ids=[],
        num_classes=2,
    )

    assert summary["n"] == 0
    assert summary["accuracy"] is None
    assert summary["macro_accuracy"] is None


def test_delta_scores_align_by_dataset_index_not_loader_order() -> None:
    current = _outputs(
        labels=[3, 3],
        predictions=[3, 3],
        indices=[20, 10],
        scores=[0.2, 0.9],
    )
    reference = _outputs(
        labels=[3, 3],
        predictions=[3, 3],
        indices=[10, 20],
        scores=[0.5, 0.4],
    )

    current_scores, reference_scores = _aligned_score_pairs(
        current, reference, class_ids=[3], max_samples=10
    )

    assert current_scores.tolist() == pytest.approx([0.2, 0.9])
    assert reference_scores.tolist() == pytest.approx([0.4, 0.5])


def test_group_classes_uses_hierarchy_metadata_and_fallback() -> None:
    groups = _group_classes(
        {
            "target_classes": [2],
            "sibling_classes": [0, 1],
        },
        num_classes=5,
    )

    assert groups["target"] == [2]
    assert groups["sibling"] == [0, 1]
    assert groups["unrelated"] == [3, 4]
    assert groups["retain"] == [0, 1, 3, 4]


def test_group_classes_rejects_overlapping_hierarchy() -> None:
    with pytest.raises(ValueError, match="overlap"):
        _group_classes(
            {
                "target_classes": [2],
                "sibling_classes": [2],
                "unrelated_classes": [0, 1],
            },
            num_classes=3,
        )


def test_candidate_contract_rejects_silent_truncation_and_overwrite() -> None:
    with pytest.raises(ValueError, match="equal length"):
        _validate_candidates(["one"], ["a.pt", "b.pt"])
    with pytest.raises(ValueError, match="unique"):
        _validate_candidates(["one", "one"], ["a.pt", "b.pt"])


def test_hierarchy_artifacts_are_machine_readable(tmp_path) -> None:
    outputs = _outputs(
        labels=[0, 1, 2],
        predictions=[0, 2, 2],
    )
    class_groups = {
        "target": [0],
        "sibling": [1],
        "unrelated": [2],
        "retain": [1, 2],
        "all": [0, 1, 2],
    }
    groups = {
        name: summarize_classification_group(outputs, ids, 3)
        for name, ids in class_groups.items()
    }

    paths = _write_hierarchy_artifacts(
        tmp_path,
        "counterfactual/rebind",
        {
            "class_names": ["zero", "one", "two"],
            "class_groups": class_groups,
            "groups": groups,
            "mia_sample_count": 1,
        },
    )

    assert _safe_artifact_name("counterfactual/rebind") == (
        "counterfactual_rebind"
    )
    assert "counterfactual_rebind" in paths["per_class_path"]
    assert (
        tmp_path
        / "hierarchy/counterfactual_rebind_group_confusion.csv"
    ).exists()
    payload = json.loads(
        (tmp_path / "hierarchy/counterfactual_rebind_groups.json").read_text()
    )
    assert payload["groups"]["target"]["accuracy"] == 1.0
    assert payload["groups"]["sibling"]["accuracy"] == 0.0
