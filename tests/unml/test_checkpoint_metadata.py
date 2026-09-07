from __future__ import annotations

import torch

from unml.model import update_checkpoint_extra


def test_update_checkpoint_extra_preserves_payload(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_config": {"adapter_type": "vision_lora"},
            "adapter_state_dict": {"weight": torch.tensor([1.0])},
            "architecture": {"trainable_parameter_count": 1},
            "extra": {"stage": "finetuned"},
        },
        path,
    )

    update_checkpoint_extra(
        str(path), {"benchmark": {"total_seconds": 3.0}}
    )

    payload = torch.load(path, map_location="cpu")
    assert payload["extra"]["stage"] == "finetuned"
    assert payload["extra"]["benchmark"]["total_seconds"] == 3.0
    assert payload["adapter_state_dict"]["weight"].item() == 1.0
