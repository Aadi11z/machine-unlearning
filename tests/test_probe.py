from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from unml import probe


class TinyDataset:
    def __init__(self) -> None:
        self.targets = [0, 1, 1, 2]
        self.images = [
            Image.new("RGB", (4, 4), color=(index * 20, 0, 0))
            for index in range(4)
        ]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.images[index], self.targets[index]


class DummyProcessor:
    size = {"shortest_edge": 4}
    crop_size = {"height": 4}
    image_mean = [0.0, 0.0, 0.0]
    image_std = [1.0, 1.0, 1.0]


class DummyTokenizer:
    def __call__(self, prompts, **_kwargs):
        count = len(prompts)
        return {
            "input_ids": torch.ones((count, 2), dtype=torch.long),
            "attention_mask": torch.ones((count, 2), dtype=torch.long),
        }


class ImageProcessor:
    def __call__(self, images, **_kwargs):
        assert images.mode == "RGB"
        return {"pixel_values": torch.zeros((1, 3, 4, 4))}


class ImageModel:
    cfg = SimpleNamespace(precision="fp32")

    def class_logits_from_text_features(self, pixel_values, class_text_features):
        assert pixel_values.shape == (1, 3, 4, 4)
        assert class_text_features.shape == (3, 2)
        return torch.tensor([[0.2, 3.0, 1.0]])


def test_resolve_probe_classes_supports_names_and_target_default() -> None:
    names = ["orchid", "rose", "tulip"]

    assert probe.resolve_probe_classes(
        class_ids=[],
        class_names_requested=["rose"],
        class_names=names,
        target_classes=[0],
    ) == [1]
    assert probe.resolve_probe_classes(
        class_ids=[],
        class_names_requested=[],
        class_names=names,
        target_classes=[0],
    ) == [0]


def test_select_probe_indices_combines_explicit_and_class_samples() -> None:
    selected = probe.select_probe_indices(
        dataset=TinyDataset(),
        explicit_indices=[3],
        class_ids=[1],
        limit_per_class=1,
    )

    assert selected == [3, 1]


def test_default_train_probe_uses_exact_forget_indices() -> None:
    assert probe.default_probe_indices(
        source="train",
        split={"forget_indices": [1, 2, 3]},
        dataset=TinyDataset(),
        limit_per_class=1,
    ) == [1, 3]
    assert probe.default_probe_indices(
        source="test",
        split={"forget_indices": [1, 2]},
        dataset=TinyDataset(),
        limit_per_class=1,
    ) == []


def test_predict_probe_image_returns_target_rank_without_true_label() -> None:
    predictor = probe.ImageCheckpointPredictor(
        model=ImageModel(),
        metadata={},
        class_names=("orchid", "rose", "tulip"),
        image_processor=ImageProcessor(),
        class_text_features=torch.zeros((3, 2)),
        device=torch.device("cpu"),
    )

    result = probe.predict_probe_image(
        predictor,
        Image.new("RGBA", (4, 4), color=(0, 0, 0, 0)),
        target_class_name="rose",
        top_k=2,
    )

    assert result["target_class_id"] == 1
    assert result["target_rank"] == 1
    assert result["top_k"][0]["class_name"] == "rose"
    assert len(result["top_k"]) == 2
    assert result["class_scores"]["rose"] > result["class_scores"]["tulip"]


def test_run_checkpoint_probe_writes_comparison_and_images(
    tmp_path, monkeypatch
) -> None:
    dataset = TinyDataset()
    split = {
        "request_name": "selective",
        "target_classes": [1],
        "sibling_classes": [0],
        "unrelated_classes": [2],
    }
    monkeypatch.setattr(
        probe,
        "load_split_metadata",
        lambda *_args: (
            split,
            SimpleNamespace(name="cifar100"),
            ["orchid", "rose", "tulip"],
        ),
    )
    monkeypatch.setattr(
        probe,
        "load_dataset_pair",
        lambda *_args: (dataset, dataset),
    )
    monkeypatch.setattr(
        probe.torch,
        "load",
        lambda *_args, **_kwargs: {
            "model_config": {"model_name": "tiny-clip"}
        },
    )
    monkeypatch.setattr(
        probe.CLIPImageProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: DummyProcessor(),
    )
    monkeypatch.setattr(
        probe.CLIPTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: DummyTokenizer(),
    )

    def fake_predict(
        checkpoint_path,
        _dataset_name,
        batch,
        _class_text_inputs,
        _device,
        _top_k,
        _expected_model_name,
    ):
        is_reference = checkpoint_path == "base.pt"
        rows = []
        for label in batch["labels"].tolist():
            predicted = label if is_reference else 2
            rows.append(
                {
                    "predicted_class_id": predicted,
                    "prediction_confidence": 0.8,
                    "true_confidence": 0.7 if is_reference else 0.2,
                    "top_k_class_ids": [predicted, label],
                    "top_k_confidences": [0.8, 0.2],
                }
            )
        return {"dataset": "cifar100"}, rows

    monkeypatch.setattr(probe, "_predict_checkpoint", fake_predict)

    result = probe.run_checkpoint_probe(
        data_dir="unused",
        split_path="unused",
        dataset_name="cifar100",
        source="test",
        indices=[1],
        class_ids=[],
        class_names_requested=[],
        limit_per_class=2,
        reference_checkpoint="base.pt",
        candidate_names=["unlearned"],
        candidate_checkpoints=["unlearned.pt"],
        output_dir=str(tmp_path),
        prompt_template="a photo of a {}",
        top_k=2,
        device_name="cpu",
        save_images=True,
        local_files_only=True,
    )

    payload = json.loads((tmp_path / "probe.json").read_text())
    assert result["sample_count"] == 1
    assert result["model_count"] == 2
    assert payload["rows"][1]["true_confidence_delta"] == pytest.approx(-0.5)
    assert payload["rows"][1]["hierarchy_group"] == "target"
    assert len(result["image_paths"]) == 1
    assert (tmp_path / "probe.md").exists()
