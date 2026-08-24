from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import nullcontext
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from .utils import transformers_offline


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
    lora_dropout: float = 0.0
    train_logit_scale: bool = True
    precision: str = "fp32"
    gradient_checkpointing: bool = False
    local_files_only: bool = False

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
        invalid_targets = set(self.lora_targets) - {"q_proj", "k_proj", "v_proj"}
        if invalid_targets:
            raise ValueError(
                f"Unsupported vision LoRA targets: {sorted(invalid_targets)}"
            )
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")


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

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank > min(base_layer.in_features, base_layer.out_features):
            raise ValueError(
                f"LoRA rank {rank} exceeds projection dimensions "
                f"{base_layer.in_features}x{base_layer.out_features}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        lora_input = self.dropout(x.to(dtype=self.lora_a.weight.dtype))
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
        self.clip = CLIPModel.from_pretrained(
            cfg.model_name,
            local_files_only=cfg.local_files_only or transformers_offline(),
        )
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
            self.clip.gradient_checkpointing_enable()
            # Checkpointed layers need a differentiable input even though the
            # pretrained vision embeddings themselves remain frozen.
            self.clip.vision_model.embeddings.register_forward_hook(
                self._require_vision_embedding_gradients
            )

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "LightweightVLM":
        return cls(cfg)

    @staticmethod
    def _require_vision_embedding_gradients(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        return output.requires_grad_(True)

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
                        dropout=self.cfg.lora_dropout,
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
            vision_output = self.clip.vision_model(pixel_values=pixel_values)
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
def _atomic_torch_save(payload: Dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


RECOVERY_CHECKPOINT_SCHEMA = "unml-recovery-v1"


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_recovery_checkpoint(
    path: str | Path,
    *,
    model: LightweightVLM,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any | None,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Atomically persist the complete epoch-boundary recovery state."""
    payload = {
        "schema": RECOVERY_CHECKPOINT_SCHEMA,
        "model_config": asdict(model.cfg),
        "adapter_state_dict": _adapter_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_state": capture_rng_state(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": dict(metrics),
        "provenance": dict(provenance),
    }
    _atomic_torch_save(payload, path)


def load_recovery_checkpoint(
    path: str | Path,
    *,
    model: LightweightVLM,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any | None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != RECOVERY_CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported recovery checkpoint schema: {path}")
    validate_adapter_state(payload["adapter_state_dict"], _adapter_state_dict(model))
    model.load_state_dict(payload["adapter_state_dict"], strict=False)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    restore_rng_state(payload["rng_state"])
    return payload




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
    _atomic_torch_save(payload, path)


def update_checkpoint_extra(path: str, updates: Dict) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    extra = payload.setdefault("extra", {})
    extra.update(updates)
    _atomic_torch_save(payload, path)


def read_checkpoint_payload(
    path: str, map_location: str | torch.device = "cpu"
) -> Dict:
    if str(path).endswith(".safetensors"):
        return _read_safetensors_payload(path)
    return torch.load(path, map_location=map_location, weights_only=False)


_SAFETENSORS_FORMAT_KEY = "format"
_SAFETENSORS_CONFIG_KEY = "model_config"
_SAFETENSORS_SCALARS_KEY = "scalar_keys"


def export_checkpoint_safetensors(payload: Dict) -> bytes:
    """Serialize a checkpoint payload without pickle execution risk."""
    import io
    import json

    from safetensors.torch import save

    state = payload["adapter_state_dict"]
    tensors = {}
    scalar_keys = []
    for key, value in state.items():
        if value.dim() == 0:
            scalar_keys.append(key)
            tensors[key] = value.detach().cpu().clone().reshape(1)
        else:
            tensors[key] = value.detach().cpu().contiguous()
    metadata = {
        _SAFETENSORS_FORMAT_KEY: "unml-adapter-v1",
        _SAFETENSORS_CONFIG_KEY: json.dumps(payload["model_config"]),
        _SAFETENSORS_SCALARS_KEY: json.dumps(scalar_keys),
    }
    return save(tensors, metadata=metadata)


def read_safetensors_metadata(data: bytes) -> Dict[str, str]:
    """Read the __metadata__ mapping from safetensors bytes without loading tensors."""
    import json
    import struct

    if len(data) < 8:
        raise ValueError("Safetensors payload is truncated")
    (header_length,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + header_length])
    metadata = header.get("__metadata__", {})
    return {str(key): str(value) for key, value in metadata.items()}


def _read_safetensors_payload(path: str) -> Dict:
    import json

    from safetensors import safe_open
    from safetensors.torch import load_file

    tensors = load_file(path)
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata() or {}
    if metadata.get(_SAFETENSORS_FORMAT_KEY) != "unml-adapter-v1":
        raise ValueError(f"Checkpoint {path!r} is not a unml adapter safetensors file")
    scalar_keys = set(json.loads(metadata.get(_SAFETENSORS_SCALARS_KEY, "[]")))
    state = {
        key: (
            value.reshape(())
            if key in scalar_keys
            else value
        )
        for key, value in tensors.items()
    }
    return {
        "model_config": json.loads(metadata[_SAFETENSORS_CONFIG_KEY]),
        "adapter_state_dict": state,
    }


_NON_STRUCTURAL_CONFIG_FIELDS = frozenset(
    {"train_logit_scale", "gradient_checkpointing", "local_files_only"}
)


def _json_comparable(value: Any) -> Any:
    """Normalize values so tuple/list storage differences do not matter."""
    import json

    return json.loads(json.dumps(value))


def validate_adapter_payload(
    payload: Dict,
    reference_cfg: ModelConfig,
    *,
    source: str = "checkpoint",
    expected_model_name: str | None = None,
    expected_adapter_state: Mapping[str, torch.Tensor] | None = None,
) -> None:
    """Validate checkpoint metadata and, optionally, its exact tensor contract.

    ``expected_adapter_state`` is intentionally a mapping rather than a model so
    callers that receive an untrusted payload can reject it without creating a
    second frozen CLIP backbone.  Supplying it also makes this a complete
    pre-mutation validation boundary for hot-swaps.
    """
    cfg_payload = payload.get("model_config")
    if not isinstance(cfg_payload, Mapping):
        raise ValueError(f"Checkpoint {source!r} is missing model_config")
    adapter_state = payload.get("adapter_state_dict")
    if not isinstance(adapter_state, Mapping) or not adapter_state:
        raise ValueError(f"Checkpoint {source!r} contains no adapter weights")

    stored_model_name = str(cfg_payload.get("model_name", ""))
    if (
        expected_model_name is not None
        and stored_model_name != expected_model_name
    ):
        raise ValueError(
            f"Checkpoint {source!r} uses {stored_model_name}, "
            f"expected {expected_model_name}"
        )

    reference = {
        field: _json_comparable(value)
        for field, value in asdict(reference_cfg).items()
    }
    mismatches = []
    missing = []
    for field, value in reference.items():
        if field in _NON_STRUCTURAL_CONFIG_FIELDS:
            continue
        if field not in cfg_payload:
            missing.append(field)
        elif _json_comparable(cfg_payload[field]) != value:
            mismatches.append(
                f"{field}: stored={cfg_payload[field]!r} != active={value!r}"
            )
    if missing:
        raise ValueError(
            f"Checkpoint {source!r} configuration lacks fields: {missing}"
        )
    if mismatches:
        raise ValueError(
            f"Checkpoint {source!r} architecture differs from the active model: "
            + "; ".join(mismatches)
        )
    if expected_adapter_state is not None:
        validate_adapter_state(adapter_state, expected_adapter_state)


def _expected_adapter_state(model: LightweightVLM) -> dict[str, torch.Tensor]:
    """Return the complete adapter keys accepted by a live model."""
    model_state = model.state_dict()
    expected_keys = {
        name for name, param in model.named_parameters() if param.requires_grad
    }
    # Checkpoints always carry this scalar, even when it was frozen during
    # training, so a hot-swap must replace it rather than retain a stale value.
    expected_keys.add("logit_scale")
    return {key: model_state[key] for key in expected_keys}


def validate_adapter_state(
    adapter_state: Mapping[str, Any],
    expected_state: Mapping[str, torch.Tensor],
) -> None:
    """Require an exact adapter tensor contract without constructing a model."""
    expected_keys = set(expected_state)
    received_keys = set(adapter_state)
    unexpected = sorted(received_keys - expected_keys)
    if unexpected:
        raise ValueError(f"Adapter state contains unexpected keys: {unexpected}")
    missing = sorted(expected_keys - received_keys)
    if missing:
        raise ValueError(f"Adapter state is missing expected keys: {missing}")

    invalid_tensors = []
    for key in sorted(expected_keys):
        value = adapter_state[key]
        expected = expected_state[key]
        if not isinstance(expected, torch.Tensor):
            raise ValueError(
                f"Expected adapter state contains a non-tensor for key: {key}"
            )
        if not isinstance(value, torch.Tensor):
            invalid_tensors.append(f"{key}: expected a tensor")
        elif value.shape != expected.shape:
            invalid_tensors.append(
                f"{key}: shape {tuple(value.shape)} != {tuple(expected.shape)}"
            )
        elif value.dtype != expected.dtype:
            invalid_tensors.append(
                f"{key}: dtype {value.dtype} != {expected.dtype}"
            )
    if invalid_tensors:
        raise ValueError(
            "Adapter state contains incompatible tensors: "
            + "; ".join(invalid_tensors)
        )


def apply_adapter_state(model: LightweightVLM, adapter_state: Mapping[str, Any]) -> None:
    """Copy trainable adapter weights onto an already-built model in place.

    The complete adapter contract is verified before any tensor is copied so a
    rejected payload cannot leave the live model partially mutated.
    """
    validate_adapter_state(adapter_state, _expected_adapter_state(model))

    model.load_state_dict(dict(adapter_state), strict=False)


def load_checkpoint(
    path: str, map_location: str | torch.device = "cpu"
) -> Tuple[LightweightVLM, Dict]:
    payload = read_checkpoint_payload(path, map_location=map_location)
    cfg = ModelConfig(**payload["model_config"])
    model = LightweightVLM.from_config(cfg)

    if "adapter_state_dict" in payload:
        adapter_state = payload["adapter_state_dict"]
        if not isinstance(adapter_state, Mapping):
            raise ValueError(f"Checkpoint {path!r} adapter_state_dict is not a mapping")
        validate_adapter_state(adapter_state, _expected_adapter_state(model))
        incompatible = model.load_state_dict(
            dict(adapter_state), strict=False
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
