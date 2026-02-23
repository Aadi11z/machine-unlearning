from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unml.utils import format_metrics, get_device, set_seed


def test_format_metrics_sorts_keys() -> None:
    out = format_metrics({"b": 2.0, "a": 1.0})
    assert out.startswith("a=1.0000")
    assert out.endswith("b=2.0000")


def test_set_seed_reproducible() -> None:
    set_seed(123)
    values_a = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))

    set_seed(123)
    values_b = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))

    assert values_a == pytest.approx(values_b)


def test_get_device_explicit_cpu() -> None:
    assert get_device("cpu").type == "cpu"


def test_get_device_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "backends", SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
    assert get_device("auto").type == "cuda"


def test_get_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "backends", SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
    assert get_device("auto").type == "cpu"
