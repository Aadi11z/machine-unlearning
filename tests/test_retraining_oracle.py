from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import unml.train as train_module
from unml.train import (
    FineTuneConfig,
    resolve_oracle_schedule,
    resolve_training_plan,
    validate_oracle_source_config,
    validate_retraining_split,
    run_finetuning,
)


def _oracle_config(**overrides) -> FineTuneConfig:
    values = {
        "data_dir": "data",
        "split_path": "split.json",
        "output_dir": "outputs",
        "dataset_name": "cifar100",
        "model_name": "clip",
        "adapter_type": "vision_lora",
        "lora_targets": ["q_proj", "v_proj"],
        "batch_size": 64,
        "gradient_accumulation_steps": 2,
        "lr": 3e-3,
        "weight_decay": 1e-4,
        "seed": 42,
        "training_mode": "retrain_oracle",
        "train_loader_key": "retain_train",
        "initial_checkpoint": "base_init.pt",
        "source_metrics_path": "finetune_metrics.json",
        "target_optimizer_steps": 100,
        "target_scheduler_steps": 100,
    }
    values.update(overrides)
    return FineTuneConfig(**values)


def test_oracle_schedule_rejects_smoke_and_preserves_source_steps() -> None:
    assert resolve_oracle_schedule(
        {
            "global_steps": 100,
            "benchmark": {"scheduler_total_steps": 120},
            "smoke_mode": False,
        }
    ) == (100, 120)

    with pytest.raises(ValueError, match="smoke"):
        resolve_oracle_schedule(
            {"global_steps": 50, "smoke_mode": True}
        )
    with pytest.raises(ValueError, match="cannot be reconstructed"):
        resolve_oracle_schedule(
            {
                "global_steps": 50,
                "smoke_mode": False,
                "config": {"max_train_steps": 50},
            }
        )


def test_retraining_split_rejects_forget_leakage() -> None:
    validate_retraining_split(
        {
            "forget_indices": [1, 2],
            "retain_train_indices": [3, 4],
        }
    )

    with pytest.raises(ValueError, match="leaks forgotten"):
        validate_retraining_split(
            {
                "forget_indices": [1, 2],
                "retain_train_indices": [2, 3],
            }
        )


def test_training_plan_cycles_retain_loader_to_match_source_schedule() -> None:
    plan = resolve_training_plan(
        loader_batches=9,
        gradient_accumulation_steps=2,
        epochs=10,
        max_train_steps=-1,
        target_optimizer_steps=52,
        target_scheduler_steps=60,
    )

    assert plan == (5, 52, 60, 11)


def test_oracle_source_contract_detects_config_and_hash_drift() -> None:
    cfg = _oracle_config()
    source = {
        "dataset": "cifar100",
        "base_checkpoint_sha256": "abc",
        "config": {
            "dataset_name": "cifar100",
            "model_name": "clip",
            "adapter_type": "vision_lora",
            "lora_targets": ["q_proj", "v_proj"],
            "batch_size": 64,
            "gradient_accumulation_steps": 2,
            "lr": 3e-3,
            "weight_decay": 1e-4,
            "seed": 42,
        },
    }

    validate_oracle_source_config(cfg, source, "abc")

    with pytest.raises(ValueError, match="differs"):
        validate_oracle_source_config(
            _oracle_config(lr=1e-3), source, "abc"
        )
    with pytest.raises(ValueError, match="hash"):
        validate_oracle_source_config(cfg, source, "different")


def test_oracle_contract_fails_before_model_startup(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"checkpoint")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "dataset": "cifar100",
                "global_steps": 10,
                "smoke_mode": False,
                "config": {
                    "dataset_name": "cifar100",
                    "model_name": "different-model",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        train_module,
        "load_split_metadata",
        lambda *_args: (
            {
                "forget_indices": [1],
                "retain_train_indices": [2],
            },
            SimpleNamespace(name="cifar100"),
            ["class"],
        ),
    )

    def fail_startup(*_args, **_kwargs):
        raise AssertionError("model processor startup should not occur")

    monkeypatch.setattr(
        train_module.CLIPImageProcessor,
        "from_pretrained",
        fail_startup,
    )
    cfg = _oracle_config(
        output_dir=str(tmp_path / "output"),
        initial_checkpoint=str(checkpoint),
        source_metrics_path=str(metrics),
        target_optimizer_steps=10,
        target_scheduler_steps=10,
    )

    with pytest.raises(ValueError, match="differs"):
        run_finetuning(cfg)
