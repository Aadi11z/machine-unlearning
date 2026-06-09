from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from unml.evaluate import (
    collect_classification_outputs,
    evaluate_classification,
)


class CountingModel:
    def __init__(self) -> None:
        self.text_encode_calls = 0

    def eval(self):
        return self

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        self.text_encode_calls += 1
        return torch.eye(input_ids.shape[0], dtype=torch.float32)

    def class_logits_from_text_features(
        self,
        pixel_values: torch.Tensor,
        class_text_features: torch.Tensor,
    ) -> torch.Tensor:
        return pixel_values @ class_text_features.t()


def test_evaluation_encodes_class_text_once_for_multiple_batches() -> None:
    model = CountingModel()
    samples = [
        {
            "pixel_values": torch.tensor([1.0, 0.0]),
            "labels": torch.tensor(0),
            "indices": torch.tensor(index),
        }
        for index in range(4)
    ]
    loader = DataLoader(samples, batch_size=2)
    class_text_inputs = {
        "input_ids": torch.ones((2, 3), dtype=torch.long),
        "attention_mask": torch.ones((2, 3), dtype=torch.long),
    }

    metrics = evaluate_classification(
        model,
        loader,
        class_text_inputs,
        torch.device("cpu"),
    )

    assert model.text_encode_calls == 1
    assert metrics["n"] == 4.0


def test_collected_outputs_include_index_aligned_predictions_and_scores() -> None:
    model = CountingModel()
    samples = [
        {
            "pixel_values": torch.tensor([1.0, 0.0]),
            "labels": torch.tensor(0),
            "indices": torch.tensor(index),
        }
        for index in (7, 3, 9)
    ]
    loader = DataLoader(samples, batch_size=2)
    class_text_inputs = {
        "input_ids": torch.ones((2, 3), dtype=torch.long),
        "attention_mask": torch.ones((2, 3), dtype=torch.long),
    }

    outputs = collect_classification_outputs(
        model,
        loader,
        class_text_inputs,
        torch.device("cpu"),
        max_samples=2,
    )

    assert outputs["indices"].tolist() == [7, 3]
    assert outputs["labels"].tolist() == [0, 0]
    assert outputs["predictions"].tolist() == [0, 0]
    assert outputs["true_scores"].shape == (2,)
