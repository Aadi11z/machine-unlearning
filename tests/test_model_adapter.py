from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

from unml.model import (
    LightweightVLM,
    LoRALinear,
    LowRankAdapter,
    ModelConfig,
    _feature_tensor,
    apply_adapter_state,
    load_checkpoint,
    read_checkpoint_payload,
    save_checkpoint,
    validate_adapter_payload,
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
    assert model.can_cache_text_features_during_training()


def test_vision_lora_gradient_checkpointing_preserves_gradients(
    monkeypatch,
) -> None:
    _patch_tiny_clip(monkeypatch, num_vision_layers=2)
    model = LightweightVLM(
        ModelConfig(
            model_name="tiny",
            adapter_type="vision_lora",
            lora_layers="all",
            gradient_checkpointing=True,
            train_logit_scale=False,
        )
    )
    model.set_train_mode()

    loss = model.encode_images(torch.randn(2, 3, 32, 32)).sum()
    loss.backward()

    nonzero_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad).item())
    ]
    assert nonzero_gradients
    assert all(".lora_" in name for name in nonzero_gradients)


def test_post_projection_adapter_does_not_cache_trainable_text_features(
    monkeypatch,
) -> None:
    _patch_tiny_clip(monkeypatch, num_vision_layers=2)
    model = LightweightVLM(
        ModelConfig(
            model_name="tiny",
            adapter_type="post_projection",
        )
    )

    assert not model.can_cache_text_features_during_training()


def test_cached_class_text_features_match_direct_class_logits(
    monkeypatch,
) -> None:
    _patch_tiny_clip(monkeypatch, num_vision_layers=2)
    model = LightweightVLM(
        ModelConfig(
            model_name="tiny",
            adapter_type="vision_lora",
            lora_layers="all",
        )
    ).eval()
    pixels = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(0, 100, (5, 8))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        text_features = model.encode_text(input_ids, attention_mask)
        direct = model.class_logits(pixels, input_ids, attention_mask)
        cached = model.class_logits_from_text_features(
            pixels, text_features
        )

    assert torch.allclose(direct, cached)


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


def _adapter_state(model: LightweightVLM) -> dict[str, torch.Tensor]:
    state = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    state["logit_scale"] = model.logit_scale.detach().clone()
    return state


def _assert_model_state_unchanged(
    model: LightweightVLM, reference: dict[str, torch.Tensor]
) -> None:
    for key, value in model.state_dict().items():
        assert torch.equal(value, reference[key])


def test_apply_adapter_state_rejects_missing_keys_before_copying(monkeypatch) -> None:
    _patch_tiny_clip(monkeypatch, num_vision_layers=2)
    model = LightweightVLM(
        ModelConfig(
            model_name="tiny",
            adapter_type="vision_lora",
            lora_layers="all",
            train_logit_scale=False,
        )
    )
    reference = copy.deepcopy(model.state_dict())
    adapter_state = _adapter_state(model)
    first_key, missing_key = list(adapter_state)[:2]
    adapter_state[first_key].add_(1.0)
    del adapter_state[missing_key]

    with pytest.raises(ValueError, match="missing expected keys"):
        apply_adapter_state(model, adapter_state)

    _assert_model_state_unchanged(model, reference)


def test_apply_adapter_state_rejects_wrong_shape_before_copying(monkeypatch) -> None:
    _patch_tiny_clip(monkeypatch, num_vision_layers=2)
    model = LightweightVLM(
        ModelConfig(
            model_name="tiny",
            adapter_type="vision_lora",
            lora_layers="all",
            train_logit_scale=False,
        )
    )
    reference = copy.deepcopy(model.state_dict())
    adapter_state = _adapter_state(model)
    first_key, malformed_key = list(adapter_state)[:2]
    adapter_state[first_key].add_(1.0)
    original = adapter_state[malformed_key]
    adapter_state[malformed_key] = torch.zeros(
        (*original.shape, 1), dtype=original.dtype
    )

    with pytest.raises(ValueError, match="incompatible tensors"):
        apply_adapter_state(model, adapter_state)

    _assert_model_state_unchanged(model, reference)


def test_apply_adapter_state_rejects_wrong_dtype_before_copying(monkeypatch) -> None:
    _patch_tiny_clip(monkeypatch, num_vision_layers=2)
    model = LightweightVLM(
        ModelConfig(
            model_name="tiny",
            adapter_type="vision_lora",
            lora_layers="all",
            train_logit_scale=False,
        )
    )
    reference = copy.deepcopy(model.state_dict())
    adapter_state = _adapter_state(model)
    first_key, malformed_key = list(adapter_state)[:2]
    adapter_state[first_key].add_(1.0)
    adapter_state[malformed_key] = adapter_state[malformed_key].to(torch.float64)

    with pytest.raises(ValueError, match="incompatible tensors"):
        apply_adapter_state(model, adapter_state)

    _assert_model_state_unchanged(model, reference)


@pytest.mark.parametrize("malformation", ["missing", "unexpected", "shape", "dtype"])
def test_validate_adapter_payload_checks_complete_adapter_contract(
    monkeypatch, malformation: str
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
    state = _adapter_state(model)
    key = next(name for name in state if name != "logit_scale")
    if malformation == "missing":
        del state[key]
        expected_error = "missing expected keys"
    elif malformation == "unexpected":
        state["foreign.weight"] = torch.zeros(1)
        expected_error = "unexpected keys"
    elif malformation == "shape":
        state[key] = torch.zeros((*state[key].shape, 1), dtype=state[key].dtype)
        expected_error = "incompatible tensors"
    else:
        state[key] = state[key].to(torch.float64)
        expected_error = "incompatible tensors"

    with pytest.raises(ValueError, match=expected_error):
        validate_adapter_payload(
            {
                "model_config": model.cfg.__dict__,
                "adapter_state_dict": state,
            },
            model.cfg,
            expected_adapter_state=_adapter_state(model),
        )


@pytest.mark.parametrize("malformation", ["missing", "shape"])
def test_load_checkpoint_rejects_incomplete_or_incompatible_adapter_state(
    tmp_path, monkeypatch, malformation: str
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
    checkpoint = tmp_path / f"bad_{malformation}.pt"
    save_checkpoint(str(checkpoint), model)
    payload = read_checkpoint_payload(str(checkpoint))
    state = payload["adapter_state_dict"]
    key = next(name for name in state if name != "logit_scale")
    if malformation == "missing":
        del state[key]
        expected_error = "missing expected keys"
    else:
        state[key] = torch.zeros((*state[key].shape, 1), dtype=state[key].dtype)
        expected_error = "incompatible tensors"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match=expected_error):
        load_checkpoint(str(checkpoint))


def test_apply_adapter_state_is_atomic_on_unexpected_keys(monkeypatch) -> None:
    _patch_tiny_clip(monkeypatch, num_vision_layers=2)
    model = LightweightVLM(
        ModelConfig(
            model_name="tiny",
            adapter_type="vision_lora",
            lora_layers="all",
            train_logit_scale=False,
        )
    )
    reference = copy.deepcopy(model.state_dict())

    with pytest.raises(ValueError, match="unexpected keys"):
        apply_adapter_state(model, {"does.not.exist.weight": torch.zeros(2)})

    _assert_model_state_unchanged(model, reference)


def test_validate_adapter_payload_reports_config_mismatches(
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
    checkpoint = tmp_path / "baseline.pt"
    save_checkpoint(str(checkpoint), model)
    payload = read_checkpoint_payload(str(checkpoint))
    reference_cfg = ModelConfig(**payload["model_config"])

    broken = copy.deepcopy(payload)
    broken["model_config"]["lora_rank"] = 99
    with pytest.raises(ValueError, match="lora_rank"):
        validate_adapter_payload(broken, reference_cfg, source="broken")

    with pytest.raises(ValueError, match="no adapter weights"):
        validate_adapter_payload(
            {"model_config": payload["model_config"]}, reference_cfg
        )

    wrong_name = copy.deepcopy(payload)
    wrong_name["model_config"]["model_name"] = "other"
    with pytest.raises(ValueError, match="expected tiny"):
        validate_adapter_payload(
            wrong_name, reference_cfg, expected_model_name="tiny"
        )
