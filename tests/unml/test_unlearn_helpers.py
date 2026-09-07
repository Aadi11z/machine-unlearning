from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

from unml.unlearn import (
    UnlearnConfig,
    _eval_snapshot,
    _kl_div,
    _sample_counterfactual,
    _validate_h_tgsd_config,
    _validate_h_tgsd_request,
    run_unlearning,
)


def test_sample_counterfactual_never_equals_true_label() -> None:
    labels = torch.tensor([0, 1, 2, 3, 4])
    rng = random.Random(0)

    y_cf = _sample_counterfactual(labels=labels, num_classes=5, rng=rng)

    assert y_cf.shape == labels.shape
    assert torch.all(y_cf != labels)
    assert int(y_cf.min().item()) >= 0
    assert int(y_cf.max().item()) < 5


def test_kl_div_is_near_zero_for_identical_logits() -> None:
    logits = torch.randn(6, 10)
    kl = _kl_div(logits, logits, temperature=1.5)
    assert float(kl.item()) == pytest.approx(0.0, abs=1e-6)


def test_eval_snapshot_keeps_train_forget_and_held_out_target_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaders = {
        name: object()
        for name in (
            "retain_train",
            "retain_val",
            "forget",
            "test_forget",
            "test_retain",
            "test_all",
        )
    }
    accuracies = {
        id(loader): accuracy
        for loader, accuracy in zip(
            loaders.values(), (0.9, 0.8, 0.1, 0.2, 0.7, 0.6)
        )
    }
    monkeypatch.setattr(
        "unml.unlearn.build_class_text_features",
        lambda *_args, **_kwargs: torch.zeros(1),
    )
    monkeypatch.setattr(
        "unml.unlearn.evaluate_classification",
        lambda _model, loader, **_kwargs: {"accuracy": accuracies[id(loader)]},
    )

    metrics = _eval_snapshot(
        SimpleNamespace(eval=lambda: None),
        loaders,
        {},
        torch.device("cpu"),
    )

    assert metrics["forget_acc"] == 0.1
    assert metrics["target_test_acc"] == 0.2


def test_run_unlearning_rejects_unsupported_method(tmp_path) -> None:
    cfg = UnlearnConfig(
        data_dir="unused",
        split_path="unused",
        finetuned_checkpoint="unused",
        output_dir=str(tmp_path),
        method="not_a_method",
    )
    with pytest.raises(ValueError):
        run_unlearning(cfg)


def test_h_tgsd_validation_rejects_invalid_request_and_config() -> None:
    with pytest.raises(ValueError, match="superclass or selective_class"):
        _validate_h_tgsd_request(
            {"request_type": "class", "target_classes": [1]}
        )
    with pytest.raises(ValueError, match="sibling classes"):
        _validate_h_tgsd_request(
            {
                "request_type": "selective_class",
                "target_classes": [1],
                "sibling_classes": [],
            }
        )
    cfg = UnlearnConfig(
        data_dir="unused",
        split_path="unused",
        finetuned_checkpoint="unused",
        output_dir="unused",
        method="h_tgsd",
        h_tgsd_text_weight=1.5,
    )
    with pytest.raises(ValueError, match="text_weight"):
        _validate_h_tgsd_config(cfg)
