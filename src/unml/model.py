from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


POST_PROJECTION_ADAPTER = "post_projection"
VISION_LORA_ADAPTER = "vision_lora"
SUPPORTED_ADAPTER_TYPES = {POST_PROJECTION_ADAPTER, VISION_LORA_ADAPTER}


def precision_context(precision: str, device: torch.device):
    if precision == "fp32":
        return nullcontext()
    if device.type not in {"cuda", "cpu"}:
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


@dataclass
class ModelConfig:
    model_name: str
    adapter_type: str = POST_PROJECTION_ADAPTER
    adapter_rank: int = 16
    adapter_alpha: float = 16.0
    lora_rank: int = 8
    lora_alpha: float = 8.0
    lora_layers: str | Sequence[int] = "all"
    lora_targets: Sequence[str] = ("q_proj", "v_proj")
    train_logit_scale: bool = True
    precision: str = "fp32"
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.adapter_type not in SUPPORTED_ADAPTER_TYPES:
            raise ValueError(
                f"Unsupported adapter_type={self.adapter_type!r}. "
                f"Choose from {sorted(SUPPORTED_ADAPTER_TYPES)}"
            )
        if self.adapter_rank <= 0 or self.lora_rank <= 0:
            raise ValueError("Adapter ranks must be positive")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be one of: fp32, fp16, bf16")
        invalid_targets = set(self.lora_targets) - {"q_proj", "v_proj"}
        if invalid_targets:
            raise ValueError(
                f"Unsupported vision LoRA targets: {sorted(invalid_targets)}"
            )


class LowRankAdapter(nn.Module):
    """Residual bottleneck adapter applied after CLIP projection."""

    def __init__(self, dim: int, rank: int, alpha: float):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_adapt = x.to(dtype=self.down.weight.dtype)
        delta = self.up(self.down(x_adapt)).to(dtype=x.dtype)
        return x + self.scale * delta


class LoRALinear(nn.Module):
    """Frozen linear projection with a trainable low-rank weight update."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank > min(base_layer.in_features, base_layer.out_features):
            raise ValueError(
                f"LoRA rank {rank} exceeds projection dimensions "
                f"{base_layer.in_features}x{base_layer.out_features}"
            )

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_a = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        lora_input = x.to(dtype=self.lora_a.weight.dtype)
        delta = self.lora_b(self.lora_a(lora_input)).to(dtype=base.dtype)
        return base + self.scale * delta


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


def _resolve_lora_layers(
    layer_selection: str | Sequence[int], num_layers: int
) -> list[int]:
    if isinstance(layer_selection, str):
        if layer_selection.strip().lower() == "all":
            return list(range(num_layers))
        try:
            selected = [
                int(item.strip())
                for item in layer_selection.split(",")
                if item.strip()
            ]
        except ValueError as exc:
            raise ValueError(
                "lora_layers must be 'all' or comma-separated layer indices"
            ) from exc
    else:
        selected = [int(index) for index in layer_selection]

    if not selected:
        raise ValueError("At least one LoRA layer must be selected")
    if len(set(selected)) != len(selected):
        raise ValueError("LoRA layer indices must be unique")
    invalid = [index for index in selected if index < 0 or index >= num_layers]
    if invalid:
        raise ValueError(
            f"LoRA layer indices {invalid} are outside [0, {num_layers - 1}]"
        )
    return selected


class LightweightVLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.clip = CLIPModel.from_pretrained(cfg.model_name)
        for param in self.clip.parameters():
            param.requires_grad = False

        self.image_adapter: nn.Module
        self.text_adapter: nn.Module
        if cfg.adapter_type == POST_PROJECTION_ADAPTER:
            embed_dim = self.clip.config.projection_dim
            self.image_adapter = LowRankAdapter(
                embed_dim, cfg.adapter_rank, cfg.adapter_alpha
            )
            self.text_adapter = LowRankAdapter(
                embed_dim, cfg.adapter_rank, cfg.adapter_alpha
            )
        else:
            self.image_adapter = nn.Identity()
            self.text_adapter = nn.Identity()
            self._inject_vision_lora()

        init_logit_scale = float(self.clip.logit_scale.detach().cpu().item())
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))
        self.logit_scale.requires_grad = cfg.train_logit_scale

        if cfg.gradient_checkpointing:
            self.clip.vision_model.gradient_checkpointing_enable()

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "LightweightVLM":
        return cls(cfg)

    def _inject_vision_lora(self) -> None:
        layers = self.clip.vision_model.encoder.layers
        selected_layers = _resolve_lora_layers(self.cfg.lora_layers, len(layers))
        for layer_index in selected_layers:
            attention = layers[layer_index].self_attn
            for target in self.cfg.lora_targets:
                projection = getattr(attention, target)
                if not isinstance(projection, nn.Linear):
                    raise TypeError(
                        f"Expected vision layer {layer_index} {target} to be Linear, "
                        f"got {type(projection)!r}"
                    )
                setattr(
                    attention,
                    target,
                    LoRALinear(
                        projection,
                        rank=self.cfg.lora_rank,
                        alpha=self.cfg.lora_alpha,
                    ),
                )

    def lora_module_names(self) -> list[str]:
        return [
            name for name, module in self.named_modules() if isinstance(module, LoRALinear)
        ]

    def architecture_summary(self) -> Dict[str, object]:
        trainable = {
            name: param.numel()
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        return {
            "adapter_type": self.cfg.adapter_type,
            "lora_projection_count": len(self.lora_module_names()),
            "lora_modules": self.lora_module_names(),
            "trainable_parameter_count": sum(trainable.values()),
            "trainable_parameters": trainable,
            "total_parameter_count": sum(
                param.numel() for param in self.parameters()
            ),
        }

    def set_train_mode(self) -> None:
        self.train()
        self.clip.eval()
        if (
            self.cfg.adapter_type == VISION_LORA_ADAPTER
            and self.cfg.gradient_checkpointing
        ):
            # Transformers activates checkpointing only while the vision encoder
            # is in training mode. CLIP ViT-B attention dropout is zero.
            self.clip.vision_model.train()

    def configure_for_unlearning(self, train_logit_scale: bool) -> None:
        self.logit_scale.requires_grad = train_logit_scale

    def can_cache_text_features_during_training(self) -> bool:
        return not any(
            param.requires_grad
            for param in (
                *self.clip.text_model.parameters(),
                *self.clip.text_projection.parameters(),
                *self.text_adapter.parameters(),
            )
        )

    def encode_images(
        self, pixel_values: torch.Tensor, return_patch_tokens: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if return_patch_tokens:
            vision_output = self.clip.vision_model(
                pixel_values=pixel_values,
                return_dict=True,
            )
            pooled = vision_output.pooler_output
            projected = self.clip.visual_projection(pooled)
            image_features = F.normalize(
                self.image_adapter(projected), dim=-1
            )
            patch_tokens = vision_output.last_hidden_state[:, 1:, :]
            return image_features, patch_tokens

        image_features = _feature_tensor(
            self.clip.get_image_features(pixel_values=pixel_values)
        )
        image_features = self.image_adapter(image_features)
        return F.normalize(image_features, dim=-1)

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        text_features = _feature_tensor(
            self.clip.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )
        )
        text_features = self.text_adapter(text_features)
        return F.normalize(text_features, dim=-1)

    def logits_from_embeddings(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=100)
        return scale * image_features @ text_features.t()

    def class_logits_from_text_features(
        self,
        pixel_values: torch.Tensor,
        class_text_features: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self.encode_images(pixel_values)
        return self.logits_from_embeddings(image_features, class_text_features)

    def class_logits(
        self,
        pixel_values: torch.Tensor,
        class_input_ids: torch.Tensor,
        class_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self.encode_images(pixel_values)
        text_features = self.encode_text(class_input_ids, class_attention_mask)
        return self.logits_from_embeddings(image_features, text_features)

    def pairwise_logits(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self.encode_images(pixel_values)
        text_features = self.encode_text(input_ids, attention_mask)
        return self.logits_from_embeddings(image_features, text_features)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (param for param in self.parameters() if param.requires_grad)


def _adapter_state_dict(model: LightweightVLM) -> Dict[str, torch.Tensor]:
    state = {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    state["logit_scale"] = model.logit_scale.detach().cpu().clone()
    return state


def save_checkpoint(
    path: str, model: LightweightVLM, extra: Dict | None = None
) -> None:
    payload = {
        "model_config": asdict(model.cfg),
        "adapter_state_dict": _adapter_state_dict(model),
        "architecture": model.architecture_summary(),
    }
    if extra:
        payload["extra"] = extra
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str, map_location: str | torch.device = "cpu"
) -> Tuple[LightweightVLM, Dict]:
    payload = torch.load(path, map_location=map_location)
    cfg = ModelConfig(**payload["model_config"])
    model = LightweightVLM.from_config(cfg)

    if "adapter_state_dict" in payload:
        incompatible = model.load_state_dict(
            payload["adapter_state_dict"], strict=False
        )
        unexpected = list(incompatible.unexpected_keys)
        if unexpected:
            raise ValueError(
                f"Checkpoint contains unexpected adapter keys: {unexpected}"
            )
    else:
        model.load_state_dict(payload["state_dict"])

    extra = payload.get("extra", {})
    return model, extra
