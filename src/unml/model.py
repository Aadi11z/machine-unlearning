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
    #   This is where it gets interesting. Instead of fine-tuning the huge CLIP model, you're training tiny adapter layers. Think of it like this:
    #   Conceptually, an adapter works in two steps:
    #   1. Compress (down projection): Take the CLIP output (say, 768 dimensions) and squeeze it down to a tiny "bottleneck" (rank=16 dimensions). This compression loses information, but captures the essential changes needed.
    #   2. Expand (up projection): Stretch that compressed signal back to the original 768 dimensions.
    #   3. Residual connection: Add this small change back to the original CLIP output: output = original + (scale × adapter_output)
    #   - With rank=16, you're training roughly 2 × (768 × 16) = 24,576 parameters per adapter, instead of millions
    
    def __init__(self, dim: int, rank: int, alpha: float):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / max(1, rank) # scale is a hyperparameter multiplier, controls how much the adapter's output influences the final result. 
        # Usually you can fix rank accroding to your compute, and then modify alpha say {32, 16, 8} which will let you modify scale {2.0: twice as much, 1.0: moderate, 0.5: less} to influence the adapter's contribution.
        
        # 'down' and 'up' are trainable linear layers.
        self.down = nn.Linear(dim, rank, bias=False) # 'down' projects 'compression' (768 -> 16)
        self.up = nn.Linear(rank, dim, bias=False) # 'up' projects 'expansion' (16 -> 768)
        nn.init.normal_(self.down.weight, std=1e-3) # down.weight initialized with small random values (std=1e-3): Start with a very weak compression
        nn.init.zeros_(self.up.weight) # up.weight initialized to zeros: Start with zero adaptation (do nothing)
        
        # At the start of training, the adapter contributes nothing (0 × anything = 0), so the model starts with pure CLIP behavior and gradually learns to adapt.
    
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
    #   In __init__:
    #   1. Load pretrained CLIP and freeze it
    #   2. Add two small adapter modules - one for images, one for text
    #   3. Optionally make logit_scale trainable
    
    #   The logit_scale is interesting: it's CLIP's temperature parameter that controls how sharp the similarity scores are. You might want to adjust it for your task, so it's optionally trainable.
    
    #   The encoding pipeline:
    #   Images/Text → CLIP (frozen, feature extraction) → Adapter (tiny learned adjustment) → Normalize
    #   For images: pixel values → CLIP vision encoder → image_adapter → L2 normalize
    #   For text: text tokens → CLIP text encoder → text_adapter → L2 normalize

    #   Logit computation:
    #   logit_scale.exp() × (image_features @ text_features.T)
    #   The exponential scales up the scores, and the matrix multiplication computes similarity between all image-text pairs.

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.clip = CLIPModel.from_pretrained(cfg.model_name)
        # Every parameter (weights and biases) has a '.requires_grad' flag that tells Pytorch whether to compute the gradients for this parameter during backprop
        # but setting it to False allows Pytorch to skips the gradient computation completely. If True then Pytorch would have to compute how much each parameter should change to reduce loss. So computation cost would explode and would either overfit or make the model performance worse.
        # The Parameters are: All the learnable weights and biases in the CLIP model
        # - In the vision transformer: attention heads, feedforward layers, normalization layer parameters
        # - In the text transformer: embedding matrices, attention weights, layer norms
        # - The projection head that maps both to the shared embedding space
        # - Basically every tensor in the model that can be updated
        
        for param in self.clip.parameters(): 
            param.requires_grad = False

        embed_dim = self.clip.config.projection_dim
        self.image_adapter = LowRankAdapter(embed_dim, cfg.adapter_rank, cfg.adapter_alpha)
        self.text_adapter = LowRankAdapter(embed_dim, cfg.adapter_rank, cfg.adapter_alpha)

        # A logit is a raw output of a NN before any final transformation
        init_logit_scale = float(self.clip.logit_scale.detach().cpu().item()) # Extract CLIP's Temperature (making it trainable, optional)
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

    def logits_from_embeddings(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor: 
        # The temperature 't' from CLIP is actually t = log(tau), where 'tau' is the actual parameter kept in log-space to be positive.
        # That's why we use '.exp()' to bring it normal but prevent it from exploding above
        scale = self.logit_scale.exp().clamp(max=100) 
        # scale = 1 → similarities unchanged, scores close together
        # scale = 10 → similarities magnified 10×, more confident predictions
        # scale = 100 → extremely sharp (almost one-hot), very confident
        
        # Similarity as dot product [-1, 1]
        return scale * image_features @ text_features.t()

    def class_logits(self, pixel_values: torch.Tensor, class_input_ids: torch.Tensor, class_attention_mask: torch.Tensor) -> torch.Tensor:
        img = self.encode_images(pixel_values)
        txt = self.encode_text(class_input_ids, class_attention_mask)
        return self.logits_from_embeddings(img, txt)

    def pairwise_logits(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
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
