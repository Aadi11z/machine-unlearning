from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return 0.5
    return float(roc_auc_score(y_true, scores))


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
    delta_scores = np.concatenate(
        [member_current - member_base, nonmember_current - nonmember_base]
    )

    auc_conf = _safe_auc(y, conf_scores)
    auc_delta = _safe_auc(y, delta_scores)
    metrics = {
        "mia_auc_confidence": auc_conf,
        "mia_auc_delta": auc_delta,
        "mia_resistance_confidence": _mia_resistance(auc_conf),
        "mia_resistance_delta": _mia_resistance(auc_delta),
    }

    loss_inputs = (
        member_loss_current,
        nonmember_loss_current,
        member_loss_base,
        nonmember_loss_base,
    )
    if not any(value is not None for value in loss_inputs):
        return metrics
    if any(value is None for value in loss_inputs):
        raise ValueError("Loss MIA requires current/base member and nonmember losses")
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
    loss_scores = np.concatenate([-member_loss_current, -nonmember_loss_current])
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
            "mia_resistance_loss_delta": _mia_resistance(auc_loss_delta),
        }
    )
    return metrics
