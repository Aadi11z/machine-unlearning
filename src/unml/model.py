from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


@dataclass
class ModelConfig:
    model_name: str
    adapter_rank: int = 16
    adapter_alpha: float = 16.0
    train_logit_scale: bool = True


class LowRankAdapter(nn.Module):
    def __init__(self, dim: int, rank: int, alpha: float):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / max(1, rank)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CLIP features may be fp16/bf16 on accelerator backends while adapters are fp32.
        # Compute in adapter dtype, then project back to the incoming feature dtype.
        x_in = x
        x_adapt = x_in.to(dtype=self.down.weight.dtype)
        delta = self.up(self.down(x_adapt))
        delta = delta.to(dtype=x_in.dtype)
        return x_in + self.scale * delta


def _feature_tensor(value: object) -> torch.Tensor:
    """Normalize CLIP feature return values across transformers versions."""
    if isinstance(value, torch.Tensor):
        return value

    pooler = getattr(value, "pooler_output", None)
    if isinstance(pooler, torch.Tensor):
        return pooler

    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]

    raise TypeError(f"Unsupported CLIP feature output type: {type(value)!r}")


class LightweightVLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.clip = CLIPModel.from_pretrained(cfg.model_name)
        for param in self.clip.parameters():
            param.requires_grad = False

        embed_dim = self.clip.config.projection_dim
        self.image_adapter = LowRankAdapter(embed_dim, cfg.adapter_rank, cfg.adapter_alpha)
        self.text_adapter = LowRankAdapter(embed_dim, cfg.adapter_rank, cfg.adapter_alpha)

        init_logit_scale = float(self.clip.logit_scale.detach().cpu().item())
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))
        self.logit_scale.requires_grad = cfg.train_logit_scale

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "LightweightVLM":
        return cls(cfg)

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_features = _feature_tensor(self.clip.get_image_features(pixel_values=pixel_values))
        image_features = self.image_adapter(image_features)
        return F.normalize(image_features, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        text_features = _feature_tensor(
            self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        )
        text_features = self.text_adapter(text_features)
        return F.normalize(text_features, dim=-1)

    def logits_from_embeddings(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=100)
        return scale * image_features @ text_features.t()

    def class_logits(
        self,
        pixel_values: torch.Tensor,
        class_input_ids: torch.Tensor,
        class_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        img = self.encode_images(pixel_values)
        txt = self.encode_text(class_input_ids, class_attention_mask)
        return self.logits_from_embeddings(img, txt)

    def pairwise_logits(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        img = self.encode_images(pixel_values)
        txt = self.encode_text(input_ids, attention_mask)
        return self.logits_from_embeddings(img, txt)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        params = [*self.image_adapter.parameters(), *self.text_adapter.parameters()]
        if self.logit_scale.requires_grad:
            params.append(self.logit_scale)
        return params


def save_checkpoint(path: str, model: LightweightVLM, extra: Dict | None = None) -> None:
    payload = {
        "model_config": asdict(model.cfg),
        "state_dict": model.state_dict(),
    }
    if extra:
        payload["extra"] = extra
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Tuple[LightweightVLM, Dict]:
    payload = torch.load(path, map_location=map_location)
    cfg = ModelConfig(**payload["model_config"])
    model = LightweightVLM.from_config(cfg)
    model.load_state_dict(payload["state_dict"])
    extra = payload.get("extra", {})
    return model, extra
