from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from unml.model import (
    ModelConfig,
    LightweightVLM,
    read_checkpoint_payload,
    save_checkpoint,
)


CLASS_NAMES = ("rose", "tulip", "woman")


def _lora_model() -> LightweightVLM:
    return LightweightVLM(
        ModelConfig(
            model_name="tiny",
            adapter_type="vision_lora",
            lora_rank=4,
            lora_alpha=4.0,
            lora_layers="all",
            lora_targets=("q_proj", "v_proj"),
            train_logit_scale=False,
        ),
    )


class _FakeProcessor:
    def __call__(self, images, return_tensors=None):
        return {"pixel_values": torch.zeros(1, 3, 32, 32)}


@pytest.fixture
def registry_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_clip_factory):
    monkeypatch.setattr(
        "unml.model.CLIPModel.from_pretrained",
        lambda *_args, **_kwargs: tiny_clip_factory(),
    )
    model = _lora_model()
    baseline_path = tmp_path / "baseline.pt"
    save_checkpoint(str(baseline_path), model)

    payload = read_checkpoint_payload(str(baseline_path))
    candidate_state = {
        key: value.clone()
        for key, value in payload["adapter_state_dict"].items()
    }
    lora_key = next(key for key in candidate_state if key.endswith("lora_b.weight"))
    candidate_state[lora_key] = candidate_state[lora_key] + 0.05
    candidate_state["logit_scale"] = candidate_state["logit_scale"] + 0.5
    candidate_path = tmp_path / "candidate.pt"
    torch.save(
        {
            "model_config": payload["model_config"],
            "adapter_state_dict": candidate_state,
        },
        candidate_path,
    )

    text_features = torch.nn.functional.normalize(
        torch.randn(3, 16, generator=torch.Generator().manual_seed(0)), dim=-1
    )

    def fake_loader(**kwargs):
        assert kwargs["checkpoint_path"] == str(baseline_path)
        return SimpleNamespace(
            model=model,
            metadata={},
            class_names=CLASS_NAMES,
            image_processor=_FakeProcessor(),
            class_text_features=text_features,
            device=torch.device("cpu"),
        )

    monkeypatch.setattr(
        "unml.probe.load_image_checkpoint_predictor", fake_loader
    )

    from interface.registry import ModelRegistry

    registry = ModelRegistry(
        baseline_checkpoint_path=str(baseline_path),
        baseline_name="Fine-tuned baseline",
        dataset_name="cifar100",
        class_names=list(CLASS_NAMES),
        prompt_template="a photo of a {}",
        device_name="cpu",
        predictor_loader=fake_loader,
    )
    image = Image.new("RGB", (32, 32), color=(120, 30, 30))
    return SimpleNamespace(
        registry=registry,
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        payload=payload,
        image=image,
    )


def test_activate_swaps_predictions_and_reset_restores(registry_env) -> None:
    env = registry_env
    registry = env.registry
    kwargs = {"target_class_name": "rose", "top_k": 2}
    baseline_prediction = registry.predict(env.image, **kwargs)
    assert registry.active_name == "Fine-tuned baseline"

    registry.activate("candidate", str(env.candidate_path))
    candidate_prediction = registry.predict(env.image, **kwargs)
    assert (
        candidate_prediction["target_probability"]
        != baseline_prediction["target_probability"]
    )

    registry.reset_to_baseline()
    assert registry.predict(env.image, **kwargs) == baseline_prediction


def test_compare_returns_pair_and_leaves_baseline_active(registry_env) -> None:
    env = registry_env
    result = env.registry.compare(
        env.image,
        candidate_name="candidate",
        candidate_checkpoint_path=str(env.candidate_path),
        target_class_name="rose",
        top_k=2,
    )
    assert result["candidate_name"] == "candidate"
    assert (
        result["baseline_prediction"]["target_probability"]
        != result["candidate_prediction"]["target_probability"]
    )
    assert env.registry.active_name == "Fine-tuned baseline"


def test_rejects_incompatible_architecture_without_mutating(
    registry_env, tmp_path
) -> None:
    env = registry_env
    mismatched = LightweightVLM(
        ModelConfig(
            model_name="tiny",
            adapter_type="vision_lora",
            lora_rank=2,
            lora_alpha=4.0,
            lora_layers="all",
            lora_targets=("q_proj", "v_proj"),
            train_logit_scale=False,
        ),
    )
    mismatched_path = tmp_path / "mismatched.pt"
    save_checkpoint(str(mismatched_path), mismatched)

    before = env.registry.predict(env.image, target_class_name="rose", top_k=2)
    with pytest.raises(ValueError, match="differs from the active model"):
        env.registry.activate("bad", str(mismatched_path))
    assert env.registry.active_name == "Fine-tuned baseline"
    after = env.registry.predict(env.image, target_class_name="rose", top_k=2)
    assert after == before


def test_rejects_unexpected_keys_before_copying(registry_env, tmp_path) -> None:
    env = registry_env
    poisoned = {
        key: value.clone()
        for key, value in env.payload["adapter_state_dict"].items()
    }
    poisoned["foreign.layer.weight"] = torch.zeros(3)
    poisoned_path = tmp_path / "poisoned.pt"
    torch.save(
        {
            "model_config": env.payload["model_config"],
            "adapter_state_dict": poisoned,
        },
        poisoned_path,
    )

    before = env.registry.predict(env.image, target_class_name="rose", top_k=2)
    with pytest.raises(ValueError, match="unexpected keys"):
        env.registry.activate("poison", str(poisoned_path))
    assert env.registry.active_name == "Fine-tuned baseline"
    after = env.registry.predict(env.image, target_class_name="rose", top_k=2)
    assert after == before


def test_baseline_prediction_cache_hits_between_compares(
    registry_env, monkeypatch
) -> None:
    env = registry_env
    registry = env.registry
    calls = {"count": 0}
    original = registry.predict

    def counting(image, **kwargs):
        calls["count"] += 1
        return original(image, **kwargs)

    monkeypatch.setattr(registry, "predict", counting)
    for _ in range(2):
        registry.compare(
            env.image,
            candidate_name="candidate",
            candidate_checkpoint_path=str(env.candidate_path),
            target_class_name="rose",
            top_k=2,
        )
    assert calls["count"] == 3


def test_cache_distinguishes_different_images(registry_env) -> None:
    env = registry_env
    other_image = Image.new("RGB", (32, 32), color=(10, 200, 90))
    first = env.registry.compare(
        env.image,
        candidate_name="candidate",
        candidate_checkpoint_path=str(env.candidate_path),
        target_class_name="rose",
        top_k=2,
    )
    second = env.registry.compare(
        other_image,
        candidate_name="candidate",
        candidate_checkpoint_path=str(env.candidate_path),
        target_class_name="rose",
        top_k=2,
    )
    assert (
        first["baseline_prediction"]
        is not second["baseline_prediction"]
    )
