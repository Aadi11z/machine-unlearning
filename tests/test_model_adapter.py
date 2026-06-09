from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

from unml.model import (
    LoRALinear,
    LowRankAdapter,
    LightweightVLM,
    ModelConfig,
    _feature_tensor,
    load_checkpoint,
    save_checkpoint,
)


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


def _tiny_clip(num_vision_layers: int = 12) -> CLIPModel:
    vision = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_vision_layers,
        num_attention_heads=4,
        image_size=32,
        patch_size=16,
        projection_dim=16,
    )
    text = CLIPTextConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=16,
        projection_dim=16,
    )
    config = CLIPConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        projection_dim=16,
    )
    return CLIPModel(config)


def _patch_tiny_clip(monkeypatch, num_vision_layers: int = 12) -> None:
    template = _tiny_clip(num_vision_layers)
    monkeypatch.setattr(
        "unml.model.CLIPModel.from_pretrained",
        lambda *_args, **_kwargs: copy.deepcopy(template),
    )


def test_lora_linear_is_identity_at_initialization() -> None:
    base = torch.nn.Linear(8, 8)
    lora = LoRALinear(base, rank=2, alpha=2.0)
    inputs = torch.randn(4, 8)

    assert torch.allclose(lora(inputs), base(inputs))


def test_full_vision_lora_injects_24_trainable_projections(monkeypatch) -> None:
    _patch_tiny_clip(monkeypatch)
    model = LightweightVLM(
        ModelConfig(
            model_name="tiny",
            adapter_type="vision_lora",
            lora_rank=4,
            lora_alpha=4.0,
            lora_layers="all",
            lora_targets=("q_proj", "v_proj"),
            train_logit_scale=False,
        )
    )

    trainable_names = {
        name for name, param in model.named_parameters() if param.requires_grad
    }

    assert model.architecture_summary()["lora_projection_count"] == 24
    assert len(model.lora_module_names()) == 24
    assert trainable_names
    assert all(
        ".lora_a." in name or ".lora_b." in name for name in trainable_names
    )
    assert not any(
        name.startswith("clip.text_model") for name in trainable_names
    )
    assert not any(".base_layer." in name for name in trainable_names)


def test_vision_lora_exposes_global_and_patch_features(monkeypatch) -> None:
    _patch_tiny_clip(monkeypatch, num_vision_layers=2)
    model = LightweightVLM(
        ModelConfig(
            model_name="tiny",
            adapter_type="vision_lora",
            lora_layers="all",
        )
    )

    global_features, patch_features = model.encode_images(
        torch.randn(2, 3, 32, 32),
        return_patch_tokens=True,
    )

    assert global_features.shape == (2, 16)
    assert patch_features.shape == (2, 4, 32)


def test_vision_lora_checkpoint_restores_only_adapter_state(
    tmp_path, monkeypatch
) -> None:
    _patch_tiny_clip(monkeypatch, num_vision_layers=2)
    model = LightweightVLM(
        ModelConfig(
            model_name="tiny",
            adapter_type="vision_lora",
            lora_layers="all",
            train_logit_scale=False,
        )
    )
    first_lora = next(
        module for module in model.modules() if isinstance(module, LoRALinear)
    )
    torch.nn.init.constant_(first_lora.lora_b.weight, 0.25)
    checkpoint = tmp_path / "lora.pt"

    save_checkpoint(str(checkpoint), model, extra={"dataset": "cifar100"})
    payload = torch.load(checkpoint, weights_only=False)
    restored, metadata = load_checkpoint(str(checkpoint))
    restored_first = next(
        module for module in restored.modules() if isinstance(module, LoRALinear)
    )

    assert metadata["dataset"] == "cifar100"
    assert payload["architecture"]["lora_projection_count"] == 4
    assert all(
        "lora_" in name or name == "logit_scale"
        for name in payload["adapter_state_dict"]
    )
    assert torch.equal(
        restored_first.lora_b.weight,
        first_lora.lora_b.weight,
    )
