from __future__ import annotations

import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

from interface.guards import (
    SlidingWindowLimiter,
    UsageTracker,
    cleanup_expired_jobs,
)


def test_limiter_blocks_after_window_is_full() -> None:
    limiter = SlidingWindowLimiter(max_events=3, window_s=60)
    assert all(limiter.allow("ip") for _ in range(3))
    assert not limiter.allow("ip")
    assert limiter.allow("other-ip")
    assert limiter.retry_after_s("other-ip") == 0.0
    assert 0 < limiter.retry_after_s("ip") <= 60


def test_limiter_window_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    limiter = SlidingWindowLimiter(max_events=1, window_s=10)
    assert limiter.allow("k")
    assert not limiter.allow("k")
    real_monotonic = time.monotonic()
    monkeypatch.setattr(
        "interface.guards.time.monotonic", lambda: real_monotonic + 11
    )
    assert limiter.allow("k")


def test_usage_tracker_enforces_monthly_budget(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "usage.json", monthly_budget=2)
    assert tracker.used() == 0
    assert tracker.try_consume()
    assert tracker.try_consume()
    assert not tracker.try_consume()
    assert tracker.used() == 2

    state = json.loads((tmp_path / "usage.json").read_text())
    month = tracker.current_month()
    state[month] = 0
    (tmp_path / "usage.json").write_text(json.dumps(state))
    assert tracker.used() == 0


def test_cleanup_expired_jobs_removes_only_old_dirs(tmp_path: Path) -> None:
    jobs_root = tmp_path / "cifar100" / "jobs"
    old_dir = jobs_root / "rose_selective_ga_kl_200"
    new_dir = jobs_root / "woman_selective_ga_kl_200"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    old_file = old_dir / "checkpoints" / "unlearn.safetensors"
    old_file.parent.mkdir()
    old_file.write_bytes(b"x")
    new_file = new_dir / "job_result.json"
    new_file.write_text("{}")

    past = time.time() - 30 * 86400
    import os

    os.utime(old_file, (past, past))
    os.utime(old_dir, (past, past))

    removed = cleanup_expired_jobs(tmp_path, max_age_days=14)
    assert removed == ["rose_selective_ga_kl_200"]
    assert not old_dir.exists()
    assert new_dir.exists()


def _tiny_clip():
    vision = CLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, image_size=32, patch_size=16, projection_dim=16,
    )
    text = CLIPTextConfig(
        vocab_size=100, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        max_position_embeddings=16, projection_dim=16,
    )
    return CLIPModel(
        CLIPConfig(text_config=text.to_dict(), vision_config=vision.to_dict(),
                   projection_dim=16)
    )


def test_safetensors_round_trip_preserves_scalars_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "unml.model.CLIPModel.from_pretrained",
        lambda *_args, **_kwargs: _tiny_clip(),
    )
    from unml.model import (
        LightweightVLM,
        ModelConfig,
        export_checkpoint_safetensors,
        read_checkpoint_payload,
    )

    cfg = ModelConfig(
        model_name="tiny",
        adapter_type="vision_lora",
        lora_rank=4,
        lora_alpha=4.0,
        train_logit_scale=False,
    )
    model = LightweightVLM(cfg)
    adapter_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    adapter_state["logit_scale"] = model.logit_scale.detach().cpu().clone()
    payload = {
        "model_config": dict(cfg.__dict__),
        "adapter_state_dict": adapter_state,
    }
    blob = export_checkpoint_safetensors(payload)
    path = tmp_path / "delta.safetensors"
    path.write_bytes(blob)

    restored = read_checkpoint_payload(str(path))
    assert restored["model_config"] == json.loads(json.dumps(payload["model_config"]))
    for key, expected in payload["adapter_state_dict"].items():
        actual = restored["adapter_state_dict"][key]
        assert torch.equal(actual, expected.cpu())
        assert actual.shape == expected.shape


def test_safetensors_delta_activates_in_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "unml.model.CLIPModel.from_pretrained",
        lambda *_args, **_kwargs: _tiny_clip(),
    )
    from unml.model import (
        LightweightVLM,
        ModelConfig,
        export_checkpoint_safetensors,
        save_checkpoint,
    )

    def build():
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

    model = build()
    baseline_path = tmp_path / "baseline.pt"
    save_checkpoint(str(baseline_path), model)

    perturbed_state = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    perturbed_state["logit_scale"] = model.logit_scale.detach().clone()
    lora_key = next(k for k in perturbed_state if k.endswith("lora_b.weight"))
    perturbed_state[lora_key] += 0.05
    delta_blob = export_checkpoint_safetensors(
        {
            "model_config": dict(model.cfg.__dict__),
            "adapter_state_dict": {
                k: v for k, v in perturbed_state.items()
            },
        }
    )
    delta_path = tmp_path / "delta.safetensors"
    delta_path.write_bytes(delta_blob)

    text_features = torch.nn.functional.normalize(
        torch.randn(3, 16, generator=torch.Generator().manual_seed(0)), dim=-1
    )

    def fake_loader(**kwargs):
        return SimpleNamespace(
            model=model,
            metadata={},
            class_names=("rose", "tulip", "woman"),
            image_processor=lambda images, return_tensors=None: {
                "pixel_values": torch.zeros(1, 3, 32, 32)
            },
            class_text_features=text_features,
            device=torch.device("cpu"),
        )

    from interface.registry import ModelRegistry

    registry = ModelRegistry(
        baseline_checkpoint_path=str(baseline_path),
        baseline_name="baseline",
        dataset_name="cifar100",
        class_names=["rose", "tulip", "woman"],
        prompt_template="a photo of a {}",
        device_name="cpu",
        predictor_loader=fake_loader,
    )
    image = _image()
    before = registry.predict(image, target_class_name="rose", top_k=2)
    registry.activate("delta", str(delta_path))
    after = registry.predict(image, target_class_name="rose", top_k=2)
    registry.reset_to_baseline()
    restored = registry.predict(image, target_class_name="rose", top_k=2)

    assert after["target_probability"] != before["target_probability"]
    assert restored == before


def _image():
    from PIL import Image

    return Image.new("RGB", (32, 32), color=(50, 90, 200))


def test_rate_limited_endpoints_return_429(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "unml.model.CLIPModel.from_pretrained",
        lambda *_args, **_kwargs: _tiny_clip(),
    )
    from unml.model import LightweightVLM, ModelConfig, save_checkpoint

    model = LightweightVLM(
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
    baseline_path = tmp_path / "finetuned_best.pt"
    save_checkpoint(str(baseline_path), model)
    catalog = _catalog_with_baseline(tmp_path, baseline_path)

    from interface.jobs import JobManager
    from interface.webapp import ProbeService, create_app

    app = create_app(
        catalog=catalog,
        job_manager=JobManager(catalog),
        probe_service=None,
        jobs_per_hour=1,
        probes_per_minute=1,
    )

    from fastapi.testclient import TestClient

    client = TestClient(app)
    first = client.post(
        "/api/unlearn",
        data={"class_ref": "rose", "method": "ga_kl", "steps": 200},
    )
    assert first.status_code == 202
    second = client.post(
        "/api/unlearn",
        data={"class_ref": "rose", "method": "ga_kl", "steps": 200},
    )
    assert second.status_code == 429
    assert "Retry-After" in second.headers


def _catalog_with_baseline(tmp_path: Path, baseline_path: Path):
    from interface.catalog import ArtifactCatalog

    candidate_dir = (
        tmp_path / "rose_selective" / "unlearn_ga_kl_200" / "checkpoints"
    )
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "unlearn_ga_kl.pt").write_bytes(b"placeholder")
    comparison_csv = (
        tmp_path / "rose_selective" / "eval_compare_x" / "comparison.csv"
    )
    comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_csv.write_text("model,target_test_acc\n")

    output_root = tmp_path / "outputs"
    output_root.mkdir(exist_ok=True)
    (output_root / "cifar100").symlink_to(tmp_path, target_is_directory=True)
    return ArtifactCatalog(
        output_root=output_root,
        baseline_checkpoint_path=baseline_path,
    )


def test_budget_exhaustion_returns_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "unml.model.CLIPModel.from_pretrained",
        lambda *_args, **_kwargs: _tiny_clip(),
    )
    from unml.model import LightweightVLM, ModelConfig, save_checkpoint

    model = LightweightVLM(
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
    baseline_path = tmp_path / "finetuned_best.pt"
    save_checkpoint(str(baseline_path), model)
    catalog = _catalog_with_baseline(tmp_path, baseline_path)

    from interface.guards import UsageTracker
    from interface.jobs import JobManager

    def failing_runner(job):
        raise AssertionError("runner must not be called when budget is spent")

    manager = JobManager(
        catalog,
        runner=failing_runner,
        usage_tracker=UsageTracker(tmp_path / "u.json", monthly_budget=1),
    )
    manager.submit(class_ref="rose", method="ga_kl", steps=120)  # consumes budget
    with pytest.raises(RuntimeError, match="budget"):
        manager.submit(class_ref="tulip", method="ga_kl", steps=120)
