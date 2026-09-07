from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from unml.data import CLIPCollator


def test_clip_collator_uses_processor_configuration_without_network() -> None:
    processor = SimpleNamespace(
        size={"shortest_edge": 32},
        crop_size={"height": 24, "width": 24},
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.25, 0.25, 0.25],
    )
    collator = CLIPCollator(image_processor=processor)
    image = Image.fromarray(np.full((40, 48, 3), 128, dtype=np.uint8))

    batch = collator([(image, 7, 19), (image, 3, 23)])

    assert batch["pixel_values"].shape == (2, 3, 24, 24)
    assert batch["labels"].tolist() == [7, 3]
    assert batch["indices"].tolist() == [19, 23]
