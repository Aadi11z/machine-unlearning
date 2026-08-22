"""Pure JSON producers for probe comparisons.

The decision thresholds and wording preserve the original probe behavior while
remaining independent of any UI implementation.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

AUTO_RETAINED_LABEL = "Choose automatically from the baseline"

METRIC_TABLE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("target_test_acc", "Still recognizes the forgotten class (lower is better)"),
    ("retained_test_acc", "Keeps all retained classes (higher is better)"),
    ("sibling_test_acc", "Keeps other classes in the same group (higher is better)"),
    ("unrelated_test_acc", "Keeps unrelated classes (higher is better)"),
    ("utility_test_all", "Accuracy across all classes"),
    ("mia_resistance_confidence", "Training-membership privacy resistance"),
    ("mia_auc_confidence", "Raw confidence membership-attack AUC"),
)


def top_k_rows(prediction: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "rank": int(row["rank"]),
            "class_id": int(row.get("class_id", -1)),
            "class_name": str(row["class_name"]),
            "probability": float(row["probability"]),
        }
        for row in prediction["top_k"]
    ]


def metrics_table_rows(metric_row: Mapping[str, Any] | None) -> list[dict[str, str]]:
    if metric_row is None:
        return []
    return [
        {"Metric": label, "Value": f"{float(metric_row[key]):.4f}"}
        for key, label in METRIC_TABLE_COLUMNS
        if key in metric_row and _is_number(metric_row[key])
    ]


def find_metric_row(
    rows: Sequence[Mapping[str, Any]], comparison_model: str
) -> Mapping[str, Any] | None:
    for row in rows:
        if str(row.get("model")) == comparison_model:
            return row
    return None


def _is_number(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return parsed == parsed  # NaN check without importing math


def class_score(prediction: Mapping[str, Any], class_name: str) -> float:
    scores = prediction.get("class_scores", {})
    if class_name in scores:
        return float(scores[class_name])
    for row in prediction["top_k"]:
        if row["class_name"] == class_name:
            return float(row["probability"])
    return 0.0


def class_rank(prediction: Mapping[str, Any], class_name: str) -> int | None:
    if class_name == prediction["target_class_name"]:
        return int(prediction["target_rank"])
    scores = prediction.get("class_scores", {})
    if class_name not in scores:
        return None
    score = float(scores[class_name])
    return int(sum(float(value) > score for value in scores.values())) + 1


def inferred_retained_label(
    prediction: Mapping[str, Any], target_class_name: str
) -> str | None:
    top_label = str(prediction["top_k"][0]["class_name"])
    return top_label if top_label != target_class_name else None


def resolve_retained_label(
    baseline_prediction: Mapping[str, Any],
    target_class_name: str,
    retained_label: str | None,
) -> str | None:
    if retained_label in {None, AUTO_RETAINED_LABEL}:
        return inferred_retained_label(baseline_prediction, target_class_name)
    return retained_label


def _metric_value(row: Mapping[str, Any] | None, key: str) -> float | None:
    if row is None or key not in row or not _is_number(row[key]):
        return None
    return float(row[key])


def evaluate_comparison(
    *,
    target_class_name: str,
    candidate_name: str,
    baseline_prediction: Mapping[str, Any],
    candidate_prediction: Mapping[str, Any],
    metric_row: Mapping[str, Any] | None = None,
    retained_label: str | None = None,
    source_context: str | None = None,
    sibling_group_noun: str = "flower",
    num_classes: int = 100,
) -> dict[str, Any]:
    """Produce the comparison verdict and cards as JSON-safe structures."""
    baseline_probability = float(baseline_prediction["target_probability"])
    candidate_probability = float(candidate_prediction["target_probability"])
    probability_delta = candidate_probability - baseline_probability
    baseline_rank = int(baseline_prediction["target_rank"])
    candidate_rank = int(candidate_prediction["target_rank"])
    retained_label = resolve_retained_label(
        baseline_prediction, target_class_name, retained_label
    )
    retained_baseline_score = 0.0
    retained_candidate_score = 0.0
    retained_delta = 0.0
    retained_baseline_rank: int | None = None
    retained_candidate_rank: int | None = None
    if retained_label is not None:
        retained_baseline_score = class_score(baseline_prediction, retained_label)
        retained_candidate_score = class_score(candidate_prediction, retained_label)
        retained_delta = retained_candidate_score - retained_baseline_score
        retained_baseline_rank = class_rank(baseline_prediction, retained_label)
        retained_candidate_rank = class_rank(candidate_prediction, retained_label)
    candidate_top = candidate_prediction["top_k"][0]

    target_probe_is_strong = baseline_rank == 1 or baseline_probability >= 0.10
    target_weakened = candidate_probability < baseline_probability
    retained_shifted = retained_label is not None and (
        retained_delta <= -0.20
        or (
            retained_baseline_rank is not None
            and retained_candidate_rank is not None
            and retained_candidate_rank > retained_baseline_rank
        )
    )
    if not target_probe_is_strong:
        verdict = "Inconclusive for target forgetting"
        summary = (
            f"The baseline gave {target_class_name} only "
            f"{baseline_probability:.2%} relative confidence (rank {baseline_rank}), "
            "so this image is not a strong standalone test of whether that class was forgotten."
        )
    elif not target_weakened:
        verdict = "Forgotten-class signal remains visible"
        summary = (
            f"The {target_class_name} score changed from "
            f"{baseline_probability:.2%} to {candidate_probability:.2%}, so this probe "
            "does not show a reduction in the forgotten-class signal."
        )
    elif retained_shifted:
        verdict = "Target weakened, but retained content shifted"
        summary = (
            f"The {target_class_name} signal fell, but "
            f"{str(retained_label)} also changed materially. Treat this as a "
            "mixed probe rather than clean evidence of selective forgetting."
        )
    elif retained_label is not None:
        verdict = "Target signal reduced with stable retained content"
        summary = (
            f"The {target_class_name} signal fell while "
            f"{retained_label} remained comparatively stable in this image-level probe."
        )
    else:
        verdict = "Target signal reduced"
        summary = (
            f"The {target_class_name} signal fell, but this image does not "
            "provide a separate retained subject for a selectivity check."
        )

    target_accuracy = _metric_value(metric_row, "target_test_acc")
    sibling_accuracy = _metric_value(metric_row, "sibling_test_acc")
    unrelated_accuracy = _metric_value(metric_row, "unrelated_test_acc")
    utility_accuracy = _metric_value(metric_row, "utility_test_all")
    metric_cards = [
        {
            "label": (
                f"{target_class_name.capitalize()} signal in this image"
                if target_class_name
                else "Forgotten-class signal in this image"
            ),
            "value": f"{candidate_probability:.2%}",
            "note": (
                f"{baseline_probability:.2%} baseline; {probability_delta:+.2%} change; "
                f"rank {baseline_rank} to {candidate_rank}"
            ),
        },
        {
            "label": (
                f"{retained_label.capitalize()} signal in this image"
                if retained_label is not None
                else "Retained-subject signal"
            ),
            "value": (
                f"{retained_candidate_score:.2%}"
                if retained_label is not None
                else "Not available"
            ),
            "note": (
                (
                    f"{retained_baseline_score:.2%} baseline; {retained_delta:+.2%} change; "
                    f"rank {retained_baseline_rank or 'N/A'} to "
                    f"{retained_candidate_rank or 'N/A'}"
                )
                if retained_label is not None
                else "Choose a retained subject or use a mixed image to inspect collateral change."
            ),
        },
        {
            "label": "Still recognizes the forgotten class",
            "value": (
                f"{target_accuracy:.1%}" if target_accuracy is not None else "N/A"
            ),
            "note": "Held-out test result; lower is better",
        },
        {
            "label": f"Keeps other {sibling_group_noun} classes",
            "value": (
                f"{sibling_accuracy:.1%}" if sibling_accuracy is not None else "N/A"
            ),
            "note": "Held-out test result; higher is better",
        },
        {
            "label": "Keeps unrelated classes",
            "value": (
                f"{unrelated_accuracy:.1%}"
                if unrelated_accuracy is not None
                else "N/A"
            ),
            "note": "Held-out test result; higher is better",
        },
        {
            "label": f"Accuracy across all {num_classes} classes",
            "value": (
                f"{utility_accuracy:.1%}" if utility_accuracy is not None else "N/A"
            ),
            "note": "Held-out test result; higher is better",
        },
    ]
    source_note = source_context or (
        "This is an external exploratory probe without a ground-truth label."
    )

    return {
        "verdict": verdict,
        "summary": summary,
        "outcome_cards": [
            {
                "title": "What changed in this image",
                "text": (
                    f"{target_class_name} moved from rank {baseline_rank} to "
                    f"{candidate_rank}. The selected checkpoint is {candidate_name}."
                ),
            },
            {
                "title": "Candidate top label",
                "text": (
                    f"{candidate_top['class_name']} at "
                    f"{float(candidate_top['probability']):.2%} relative confidence."
                ),
            },
        ],
        "metric_cards": metric_cards,
        "source_note": source_note,
        "note": (
            "Relative confidence is the model score after normalizing across the "
            f"fixed {num_classes} labels; it is not a calibrated probability. "
            "The held-out test metrics are the evidence for experiment-level claims."
        ),
    }
