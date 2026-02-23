from __future__ import annotations

import random

import pytest
import torch

from unml.unlearn import UnlearnConfig, _kl_div, _sample_counterfactual, run_unlearning


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
