from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import train_canonical_baseline


def test_final_fit_publishes_explicit_final_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "baseline"
    checkpoint = output_dir / "checkpoints" / "finetuned_final.pt"
    metrics_path = output_dir / "metrics" / "finetune_metrics.json"
    monkeypatch.setattr(
        train_canonical_baseline,
        "parse_args",
        lambda: SimpleNamespace(
            split_path=tmp_path / "split.json",
            run_id="baseline-v2",
            data_dir=tmp_path / "data",
            output_root=tmp_path / "outputs",
            config=tmp_path / "parameters.yaml",
            device="cpu",
            precision="fp32",
            epochs=2,
            max_train_steps=None,
            offline=True,
            allow_existing=False,
            final_fit=True,
        ),
    )
    monkeypatch.setattr(
        train_canonical_baseline,
        "load_canonical_cifar100_split",
        lambda _path: {"split_id": "canonical", "digest": "digest"},
    )
    monkeypatch.setattr(
        train_canonical_baseline,
        "load_runtime_config",
        lambda _path: {"canonical": {"max_train_steps": -1}},
    )
    monkeypatch.setattr(
        train_canonical_baseline,
        "resolve_baseline_paths",
        lambda *_args, **_kwargs: SimpleNamespace(
            development=tmp_path / "development", final_fit=output_dir
        ),
    )
    captured_config = []

    def run_final_fit(cfg):
        captured_config.append(cfg)
        checkpoint.parent.mkdir(parents=True)
        metrics_path.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"final checkpoint")
        metrics_path.write_text(
            json.dumps({"final_metrics": {"test_all_acc": 0.5}})
        )
        return {
            "final_checkpoint": str(checkpoint),
            "metrics_path": str(metrics_path),
        }

    monkeypatch.setattr(
        train_canonical_baseline,
        "run_finetuning",
        run_final_fit,
    )
    monkeypatch.setattr(
        train_canonical_baseline,
        "read_checkpoint_payload",
        lambda _path: {
            "model_config": {"model_name": "clip"},
            "extra": {"prompt_contract": {"digest": "prompt"}},
        },
    )
    captured_manifest = []
    monkeypatch.setattr(
        train_canonical_baseline,
        "build_baseline_manifest",
        lambda **kwargs: captured_manifest.append(kwargs) or {"schema": "test"},
    )
    monkeypatch.setattr(
        train_canonical_baseline,
        "write_baseline_manifest",
        lambda path, manifest: path,
    )

    train_canonical_baseline.main()

    assert len(captured_config) == 1
    config = captured_config[0]
    assert config.canonical_final_fit is True
    assert config.evaluate_test is True
    assert config.max_train_steps == -1
    assert captured_manifest[0]["checkpoints"] == {
        "final_checkpoint": checkpoint
    }
