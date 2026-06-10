from __future__ import annotations

import numpy as np
import pytest

from unml.attacks import _mia_metrics, _safe_auc


def test_safe_auc_single_class_defaults_to_half() -> None:
    y_true = np.ones(8)
    scores = np.linspace(0.1, 0.9, 8)
    assert _safe_auc(y_true, scores) == pytest.approx(0.5)


def test_mia_metrics_are_bounded_and_informative() -> None:
    member_current = np.array([0.95, 0.90, 0.92, 0.88])
    nonmember_current = np.array([0.45, 0.40, 0.50, 0.55])

    member_base = np.array([0.60, 0.62, 0.58, 0.59])
    nonmember_base = np.array([0.50, 0.52, 0.48, 0.47])

    metrics = _mia_metrics(
        member_current=member_current,
        nonmember_current=nonmember_current,
        member_base=member_base,
        nonmember_base=nonmember_base,
        member_loss_current=np.array([0.05, 0.10, 0.08, 0.12]),
        nonmember_loss_current=np.array([0.80, 0.90, 0.70, 0.75]),
        member_loss_base=np.array([0.50, 0.55, 0.48, 0.52]),
        nonmember_loss_base=np.array([0.70, 0.72, 0.68, 0.74]),
    )

    assert 0.0 <= metrics["mia_auc_confidence"] <= 1.0
    assert 0.0 <= metrics["mia_auc_delta"] <= 1.0
    assert 0.0 <= metrics["mia_resistance_confidence"] <= 1.0
    assert 0.0 <= metrics["mia_resistance_delta"] <= 1.0
    assert 0.0 <= metrics["mia_auc_loss"] <= 1.0
    assert 0.0 <= metrics["mia_auc_loss_delta"] <= 1.0
    assert 0.0 <= metrics["mia_resistance_loss"] <= 1.0
    assert 0.0 <= metrics["mia_resistance_loss_delta"] <= 1.0

    # Designed so members are much more separable than non-members.
    assert metrics["mia_auc_confidence"] > 0.5
    assert metrics["mia_auc_loss"] == pytest.approx(1.0)
    assert metrics["mia_auc_loss_delta"] == pytest.approx(1.0)


def test_loss_mia_requires_complete_equal_length_inputs() -> None:
    values = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="requires"):
        _mia_metrics(
            values,
            values,
            values,
            values,
            member_loss_current=values,
        )
    with pytest.raises(ValueError, match="equal lengths"):
        _mia_metrics(
            values,
            values,
            values,
            values,
            member_loss_current=values,
            nonmember_loss_current=values,
            member_loss_base=values,
            nonmember_loss_base=np.array([0.1]),
        )
