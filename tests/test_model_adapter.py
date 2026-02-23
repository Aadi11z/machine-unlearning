from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unml.model import LowRankAdapter, _feature_tensor


def test_low_rank_adapter_is_identity_at_init() -> None:
    adapter = LowRankAdapter(dim=8, rank=2, alpha=2.0)
    x = torch.randn(4, 8)
    y = adapter(x)
    assert torch.allclose(x, y, atol=1e-6)


def test_low_rank_adapter_preserves_shape() -> None:
    adapter = LowRankAdapter(dim=16, rank=4, alpha=8.0)
    x = torch.randn(3, 16)
    y = adapter(x)
    assert y.shape == x.shape


def test_low_rank_adapter_handles_mixed_input_dtype() -> None:
    adapter = LowRankAdapter(dim=8, rank=2, alpha=2.0)
    x = torch.randn(4, 8).to(torch.float16)
    y = adapter(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_feature_tensor_accepts_tensor_output() -> None:
    x = torch.randn(2, 8)
    assert _feature_tensor(x) is x


def test_feature_tensor_extracts_pooler_output() -> None:
    pooled = torch.randn(2, 8)
    out = SimpleNamespace(pooler_output=pooled)
    assert _feature_tensor(out) is pooled


def test_feature_tensor_extracts_tuple_first_tensor() -> None:
    first = torch.randn(2, 8)
    assert _feature_tensor((first, "ignored")) is first


def test_feature_tensor_raises_on_unknown_output_type() -> None:
    with pytest.raises(TypeError):
        _feature_tensor({"pooler_output": torch.randn(2, 8)})
