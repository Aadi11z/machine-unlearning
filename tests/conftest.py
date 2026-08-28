from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Keep test imports lightweight and deterministic.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture
def tiny_clip_factory():
    """Build fresh two-layer CLIP test doubles for interface tests."""
    from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

    def build():
        vision = CLIPVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
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
        return CLIPModel(
            CLIPConfig(
                text_config=text.to_dict(),
                vision_config=vision.to_dict(),
                projection_dim=16,
            )
        )

    return build
