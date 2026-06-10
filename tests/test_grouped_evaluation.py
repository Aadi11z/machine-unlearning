from __future__ import annotations

import json

import numpy as np
import pytest

from unml.attacks import (
    _aligned_value_pairs,
    _aligned_score_pairs,
    _group_classes,
    _oracle_relative_distances,
    _run_metadata_columns,
    _safe_artifact_name,
    _validate_candidates,
    _write_hierarchy_artifacts,
    _write_oracle_artifacts,
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


def test_losses_align_by_dataset_index_not_loader_order() -> None:
    current = _outputs(
        labels=[3, 3],
        predictions=[3, 3],
        indices=[20, 10],
    )
    reference = _outputs(
        labels=[3, 3],
        predictions=[3, 3],
        indices=[10, 20],
    )
    current["sample_losses"] = np.asarray([1.2, 0.1], dtype=np.float32)
    reference["sample_losses"] = np.asarray([0.4, 0.8], dtype=np.float32)

    current_losses, reference_losses = _aligned_value_pairs(
        current,
        reference,
        class_ids=[3],
        max_samples=10,
        value_key="sample_losses",
    )

    assert current_losses.tolist() == pytest.approx([1.2, 0.1])
    assert reference_losses.tolist() == pytest.approx([0.8, 0.4])


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


def test_run_metadata_columns_expose_paper_provenance(tmp_path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")

    columns = _run_metadata_columns(
        {
            "provenance": {
                "git_commit": "abc123",
                "gpu_name": "NVIDIA A100-SXM4-80GB",
                "cuda_version": "12.4",
            },
            "runtime": {
                "total_seconds": 12.5,
                "peak_gpu_memory_mb": 2048.0,
            },
            "unlearning_config": {"seed": 123},
            "architecture": {"trainable_parameter_count": 4096},
        },
        str(checkpoint),
        dataset_name="cifar100",
        request_name="rose_selective",
    )

    assert columns["seed"] == 123
    assert columns["gpu_name"] == "NVIDIA A100-SXM4-80GB"
    assert columns["runtime_seconds"] == pytest.approx(12.5)
    assert columns["checkpoint_size_bytes"] == len(b"checkpoint")


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


def _rich_outputs(
    indices: list[int],
    labels: list[int],
    probabilities: list[list[float]],
    embeddings: list[list[float]],
) -> dict:
    outputs = _outputs(labels, labels, indices=indices)
    outputs["probabilities"] = np.asarray(probabilities, dtype=np.float32)
    outputs["embeddings"] = np.asarray(embeddings, dtype=np.float32)
    return outputs


def test_oracle_distances_align_by_index_and_preserve_empty_groups() -> None:
    oracle = _rich_outputs(
        indices=[10, 20],
        labels=[0, 2],
        probabilities=[[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    candidate = _rich_outputs(
        indices=[20, 10],
        labels=[2, 0],
        probabilities=[[0.2, 0.1, 0.7], [0.7, 0.2, 0.1]],
        embeddings=[[0.0, 1.0], [1.0, 0.0]],
    )
    groups = {
        "target": [0],
        "sibling": [],
        "unrelated": [1, 2],
        "retain": [1, 2],
        "all": [0, 1, 2],
    }

    metrics, per_sample = _oracle_relative_distances(
        candidate, oracle, groups
    )

    assert metrics["oracle_prediction_kl_target"] > 0
    assert metrics["oracle_embedding_cosine_target"] == pytest.approx(0.0)
    assert metrics["oracle_prediction_kl_sibling"] is None
    assert metrics["oracle_distance_sibling_n"] == 0
    assert per_sample["index"].tolist() == [10, 20]


def test_oracle_self_distance_is_zero_and_artifacts_are_written(
    tmp_path,
) -> None:
    outputs = _rich_outputs(
        indices=[1, 2, 3],
        labels=[0, 1, 2],
        probabilities=[
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )
    groups = {
        "target": [0],
        "sibling": [1],
        "unrelated": [2],
        "retain": [1, 2],
        "all": [0, 1, 2],
    }

    metrics, per_sample = _oracle_relative_distances(
        outputs, outputs, groups
    )
    paths = _write_oracle_artifacts(
        tmp_path, "retrain/oracle", metrics, per_sample
    )

    assert metrics["oracle_prediction_kl_all"] == pytest.approx(0.0)
    assert metrics["oracle_embedding_cosine_all"] == pytest.approx(0.0)
    assert (
        tmp_path / "oracle_reference/retrain_oracle_summary.json"
    ).exists()
    payload = json.loads(open(paths["summary_path"]).read())
    assert payload["prediction_distance"] == "KL(oracle || candidate)"


def test_oracle_distances_reject_label_mismatch_after_alignment() -> None:
    oracle = _rich_outputs(
        [1], [0], [[0.9, 0.1]], [[1.0, 0.0]]
    )
    candidate = _rich_outputs(
        [1], [1], [[0.1, 0.9]], [[0.0, 1.0]]
    )
    groups = {
        "target": [0],
        "sibling": [],
        "unrelated": [1],
        "retain": [1],
        "all": [0, 1],
    }

    with pytest.raises(ValueError, match="labels disagree"):
        _oracle_relative_distances(candidate, oracle, groups)
