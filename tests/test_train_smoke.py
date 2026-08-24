from __future__ import annotations

import torch

from unml.train import _evaluate_all


class EvalModel:
    def eval(self):
        return self

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.eye(input_ids.shape[0], dtype=torch.float32)

    def class_logits_from_text_features(
        self,
        pixel_values: torch.Tensor,
        class_text_features: torch.Tensor,
    ) -> torch.Tensor:
        return pixel_values @ class_text_features.t()


def _batch(label: int, index: int) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.tensor([[1.0, 0.0]]),
        "labels": torch.tensor([label]),
        "indices": torch.tensor([index]),
    }


def test_smoke_evaluation_limits_every_loader() -> None:
    loaders = {
        "retain_val": [_batch(0, 0), _batch(0, 1)],
        "test_all": [_batch(0, 0), _batch(0, 1)],
        "test_retain": [_batch(0, 0), _batch(0, 1)],
        "forget": [_batch(0, 0), _batch(0, 1)],
    }
    class_text_inputs = {
        "input_ids": torch.ones((2, 3), dtype=torch.long),
        "attention_mask": torch.ones((2, 3), dtype=torch.long),
    }

    metrics = _evaluate_all(
        EvalModel(),
        loaders,
        class_text_inputs,
        torch.device("cpu"),
        max_batches=1,
    )

    assert metrics["retain_val_acc"] == 1.0
    assert metrics["test_all_acc"] == 1.0

def test_pilot_evaluation_does_not_touch_test_loaders() -> None:
    loaders = {"retain_val": [_batch(0, 0)]}
    class_text_inputs = {
        "input_ids": torch.ones((2, 3), dtype=torch.long),
        "attention_mask": torch.ones((2, 3), dtype=torch.long),
    }

    metrics = _evaluate_all(
        EvalModel(),
        loaders,
        class_text_inputs,
        torch.device("cpu"),
        evaluate_test=False,
    )

    assert metrics["retain_val_acc"] == 1.0
    assert metrics["retain_val_macro_accuracy"] == 1.0
    assert metrics["retain_val_loss"] > 0.0
