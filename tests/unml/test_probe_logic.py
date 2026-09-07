from __future__ import annotations

import pytest

from interface.probe_logic import (
    AUTO_RETAINED_LABEL,
    class_rank,
    class_score,
    evaluate_comparison,
    find_metric_row,
    inferred_retained_label,
    metrics_table_rows,
    resolve_retained_label,
    top_k_rows,
)


def make_prediction(
    *,
    target_class_name: str,
    target_probability: float,
    target_rank: int,
    top_k: list[tuple[str, float]],
    class_scores: dict[str, float],
) -> dict:
    return {
        "top_k": [
            {"rank": rank, "class_id": rank - 1, "class_name": name, "probability": prob}
            for rank, (name, prob) in enumerate(top_k, start=1)
        ],
        "target_class_name": target_class_name,
        "target_probability": target_probability,
        "target_rank": target_rank,
        "class_scores": class_scores,
    }


def rose_baseline() -> dict:
    return make_prediction(
        target_class_name="rose",
        target_probability=0.45,
        target_rank=1,
        top_k=[("rose", 0.45), ("woman", 0.30), ("tulip", 0.10)],
        class_scores={"rose": 0.45, "woman": 0.30, "tulip": 0.10},
    )


def candidate(
    target_probability: float,
    target_rank: int,
    woman_probability: float = 0.29,
) -> dict:
    return make_prediction(
        target_class_name="rose",
        target_probability=target_probability,
        target_rank=target_rank,
        top_k=[("woman", woman_probability), ("tulip", 0.12)],
        class_scores={"rose": target_probability, "woman": woman_probability},
    )


def test_clean_selective_forgetting_verdict() -> None:
    result = evaluate_comparison(
        target_class_name="rose",
        candidate_name="H-TGSD",
        baseline_prediction=rose_baseline(),
        candidate_prediction=candidate(0.02, 40),
        retained_label="woman",
        metric_row={"target_test_acc": 0.05, "sibling_test_acc": 0.80,
                    "unrelated_test_acc": 0.85, "utility_test_all": 0.84},
    )
    assert result["verdict"] == "Target signal reduced with stable retained content"
    assert result["metric_cards"][0]["value"] == "2.00%"
    assert result["metric_cards"][0]["note"] == (
        "45.00% baseline; -43.00% change; rank 1 to 40"
    )
    assert result["metric_cards"][2]["value"] == "5.0%"
    assert result["metric_cards"][3]["label"] == "Keeps other flower classes"


def test_inconclusive_when_baseline_probe_is_weak() -> None:
    weak_baseline = make_prediction(
        target_class_name="rose",
        target_probability=0.05,
        target_rank=3,
        top_k=[("woman", 0.50), ("tulip", 0.30), ("rose", 0.05)],
        class_scores={"rose": 0.05, "woman": 0.50, "tulip": 0.30},
    )
    result = evaluate_comparison(
        target_class_name="rose",
        candidate_name="cand",
        baseline_prediction=weak_baseline,
        candidate_prediction=candidate(0.01, 60),
    )
    assert result["verdict"] == "Inconclusive for target forgetting"
    assert "rank 3" in result["summary"]


def test_forgotten_signal_remains_visible() -> None:
    result = evaluate_comparison(
        target_class_name="rose",
        candidate_name="ga_kl",
        baseline_prediction=rose_baseline(),
        candidate_prediction=candidate(0.60, 1),
    )
    assert result["verdict"] == "Forgotten-class signal remains visible"


def test_retained_shift_reports_mixed_probe() -> None:
    result = evaluate_comparison(
        target_class_name="rose",
        candidate_name="cand",
        baseline_prediction=rose_baseline(),
        candidate_prediction=candidate(0.02, 40, woman_probability=0.03),
        retained_label="woman",
    )
    assert result["verdict"] == "Target weakened, but retained content shifted"
    assert "-27.00% change" in result["metric_cards"][1]["note"]


def test_no_retained_subject_branch() -> None:
    baseline = rose_baseline()
    no_subject = make_prediction(
        target_class_name="rose",
        target_probability=0.45,
        target_rank=1,
        top_k=[("rose", 0.45), ("tulip", 0.30)],
        class_scores={"rose": 0.45, "tulip": 0.30},
    )
    assert inferred_retained_label(baseline, "rose") is None
    assert inferred_retained_label(no_subject, "rose") is None
    result = evaluate_comparison(
        target_class_name="rose",
        candidate_name="cand",
        baseline_prediction=no_subject,
        candidate_prediction=make_prediction(
            target_class_name="rose",
            target_probability=0.01,
            target_rank=30,
            top_k=[("tulip", 0.40)],
            class_scores={"rose": 0.01, "tulip": 0.40},
        ),
    )
    assert result["verdict"] == "Target signal reduced"
    assert result["metric_cards"][1]["value"] == "Not available"
    assert "retained subject" in result["summary"]


def test_retained_label_resolution() -> None:
    baseline = rose_baseline()
    mixed_baseline = make_prediction(
        target_class_name="rose",
        target_probability=0.45,
        target_rank=2,
        top_k=[("woman", 0.50), ("rose", 0.45)],
        class_scores={"woman": 0.50, "rose": 0.45},
    )
    assert resolve_retained_label(baseline, "rose", None) is None
    assert resolve_retained_label(mixed_baseline, "rose", None) == "woman"
    assert resolve_retained_label(baseline, "rose", AUTO_RETAINED_LABEL) is None
    assert resolve_retained_label(baseline, "rose", "tulip") == "tulip"


def test_class_score_and_rank_helpers() -> None:
    prediction = rose_baseline()
    assert class_score(prediction, "rose") == pytest.approx(0.45)
    assert class_score(prediction, "missing") == 0.0
    assert class_rank(prediction, "rose") == 1
    assert class_rank(prediction, "tulip") == 3
    assert class_rank(prediction, "absent") is None


def test_metric_cards_handle_missing_row() -> None:
    result = evaluate_comparison(
        target_class_name="rose",
        candidate_name="cand",
        baseline_prediction=rose_baseline(),
        candidate_prediction=candidate(0.02, 40),
        metric_row=None,
    )
    for card in result["metric_cards"][2:]:
        assert card["value"] == "N/A"


def test_metrics_table_rows_formatting() -> None:
    rows = metrics_table_rows(
        {"target_test_acc": 0.1234, "sibling_test_acc": "bad", "utility_test_all": None}
    )
    assert rows[0] == {
        "Metric": "Still recognizes the forgotten class (lower is better)",
        "Value": "0.1234",
    }
    assert len(rows) == 1
    assert metrics_table_rows(None) == []


def test_metrics_table_labels_all_retained_accuracy_truthfully() -> None:
    rows = metrics_table_rows({"retained_test_acc": 0.8125})
    assert rows == [
        {
            "Metric": "Keeps all retained classes (higher is better)",
            "Value": "0.8125",
        }
    ]


def test_find_metric_row_matches_model_column() -> None:
    rows = [{"model": "finetuned"}, {"model": "h_tgsd", "x": 1}]
    assert find_metric_row(rows, "h_tgsd")["x"] == 1
    assert find_metric_row(rows, "missing") is None


def test_top_k_rows_shape() -> None:
    rows = top_k_rows(rose_baseline())
    assert rows[0] == {
        "rank": 1,
        "class_id": 0,
        "class_name": "rose",
        "probability": 0.45,
    }
