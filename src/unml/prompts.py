"""Versioned CLIP text-prototype contracts.

The canonical CIFAR-100 baseline uses OpenAI's published CIFAR-100 prompt
templates.  This module keeps the template order, feature aggregation, and
contract digest in one place so that training, evaluation, unlearning, and
serving can prove that they used identical text prototypes.

The helpers intentionally do not alter the legacy single-template pipeline.
Callers must opt in to this contract rather than silently mixing the two
artifact families.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch
import torch.nn.functional as F


CIFAR100_PROMPT_CONTRACT_VERSION = "openai_cifar100_v1"

# Published by OpenAI for CIFAR-100:
# https://github.com/openai/CLIP/blob/main/data/prompts.md
# Keep the exact source order: it is part of the prototype contract and its
# digest.
CIFAR100_OPENAI_TEMPLATES: tuple[str, ...] = (
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a black and white photo of the {}.",
    "a low contrast photo of the {}.",
    "a high contrast photo of the {}.",
    "a bad photo of the {}.",
    "a good photo of the {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
)


class _Tokenizer(Protocol):
    def __call__(self, texts: Sequence[str], **kwargs: Any) -> Any: ...


class _TextEncoder(Protocol):
    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor: ...


def _normalized_text(value: str) -> str:
    """Return canonical Unicode text without changing prompt semantics."""
    return unicodedata.normalize("NFC", value)


@dataclass(frozen=True)
class PromptEnsembleContract:
    """An immutable, digestible ordered text-prompt contract."""

    dataset_name: str
    version: str
    templates: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized_dataset = self.dataset_name.strip().lower()
        if not normalized_dataset:
            raise ValueError("Prompt contract dataset_name cannot be empty")
        if not self.version.strip():
            raise ValueError("Prompt contract version cannot be empty")
        if not self.templates:
            raise ValueError("Prompt contract must contain at least one template")

        normalized_templates = tuple(
            _normalized_text(str(template)) for template in self.templates
        )
        for template in normalized_templates:
            if template.count("{}") != 1:
                raise ValueError(
                    "Each prompt template must contain exactly one '{}' placeholder"
                )
        if len(set(normalized_templates)) != len(normalized_templates):
            raise ValueError("Prompt contract templates must be unique")

        object.__setattr__(self, "dataset_name", normalized_dataset)
        object.__setattr__(self, "version", self.version.strip())
        object.__setattr__(self, "templates", normalized_templates)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "templates": list(self.templates),
            "version": self.version,
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


CIFAR100_PROMPT_CONTRACT = PromptEnsembleContract(
    dataset_name="cifar100",
    version=CIFAR100_PROMPT_CONTRACT_VERSION,
    templates=CIFAR100_OPENAI_TEMPLATES,
)


def resolve_prompt_contract(dataset_name: str) -> PromptEnsembleContract:
    """Resolve the sole canonical prompt contract for a supported dataset."""
    normalized = dataset_name.strip().lower().replace("-", "").replace("_", "")
    if normalized != "cifar100":
        raise ValueError(
            "No canonical prompt ensemble is defined for "
            f"dataset={dataset_name!r}; legacy single-template artifacts are incompatible"
        )
    return CIFAR100_PROMPT_CONTRACT


def render_class_prompts(
    class_names: Sequence[str], contract: PromptEnsembleContract
) -> tuple[str, ...]:
    """Render prompts in stable class-major, template-minor order."""
    if not class_names:
        raise ValueError("At least one class name is required")
    normalized_names = tuple(_normalized_text(str(name)) for name in class_names)
    if any(not name.strip() for name in normalized_names):
        raise ValueError("Class names cannot be empty")
    return tuple(
        template.format(class_name)
        for class_name in normalized_names
        for template in contract.templates
    )


@dataclass(frozen=True)
class PromptEnsembleInputs:
    """Tokenized class-major prompt rows plus the contract that produced them."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    class_count: int
    template_count: int
    contract: PromptEnsembleContract

    def __post_init__(self) -> None:
        expected_rows = self.class_count * self.template_count
        if self.class_count < 1 or self.template_count < 1:
            raise ValueError("Prompt ensemble counts must be positive")
        if self.input_ids.ndim != 2 or self.attention_mask.ndim != 2:
            raise ValueError("Tokenized prompt tensors must be rank two")
        if self.input_ids.shape != self.attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have identical shapes")
        if self.input_ids.shape[0] != expected_rows:
            raise ValueError(
                "Tokenized prompt rows do not match class_count * template_count"
            )


def tokenize_prompt_ensemble(
    tokenizer: _Tokenizer,
    class_names: Sequence[str],
    contract: PromptEnsembleContract,
    *,
    max_length: int = 32,
) -> PromptEnsembleInputs:
    """Tokenize the rendered prompt matrix without changing its row ordering."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    prompts = render_class_prompts(class_names, contract)
    tokens = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return PromptEnsembleInputs(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        class_count=len(class_names),
        template_count=len(contract.templates),
        contract=contract,
    )


def aggregate_prompt_features(
    prompt_features: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize each prompt, average per class, then normalize each prototype.

    ``prompt_features`` must have shape ``[classes, templates, features]``.
    The two normalizations prevent individual prompt vector magnitudes from
    changing their weight in the class prototype, and leave prototypes on the
    cosine-similarity unit sphere expected by CLIP.
    """
    if prompt_features.ndim != 3:
        raise ValueError(
            "Prompt features must have shape [classes, templates, features]"
        )
    if prompt_features.shape[0] < 1 or prompt_features.shape[1] < 1:
        raise ValueError("Prompt features require at least one class and template")
    if prompt_features.shape[2] < 1:
        raise ValueError("Prompt features require a non-empty feature dimension")
    if not torch.is_floating_point(prompt_features):
        raise ValueError("Prompt features must use a floating point dtype")
    if not torch.isfinite(prompt_features).all():
        raise ValueError("Prompt features must be finite")
    template_norms = torch.linalg.vector_norm(prompt_features, dim=-1)
    if torch.any(template_norms <= eps):
        raise ValueError("Prompt features must not contain zero-norm vectors")

    normalized = F.normalize(prompt_features, dim=-1, eps=eps)
    averaged = normalized.mean(dim=1)
    prototype_norms = torch.linalg.vector_norm(averaged, dim=-1)
    if torch.any(prototype_norms <= eps):
        raise ValueError(
            "Averaged prompt features produced a zero-norm class prototype"
        )
    return F.normalize(averaged, dim=-1, eps=eps)


def encode_prompt_ensemble(
    model: _TextEncoder,
    prompt_inputs: PromptEnsembleInputs,
    device: torch.device,
) -> torch.Tensor:
    """Encode and aggregate one contract into ``[classes, feature_dim]``."""
    flat_features = model.encode_text(
        prompt_inputs.input_ids.to(device),
        prompt_inputs.attention_mask.to(device),
    )
    expected_rows = prompt_inputs.class_count * prompt_inputs.template_count
    if flat_features.ndim != 2 or flat_features.shape[0] != expected_rows:
        raise ValueError(
            "Text encoder output does not match the prompt ensemble row count"
        )
    return aggregate_prompt_features(
        flat_features.reshape(
            prompt_inputs.class_count,
            prompt_inputs.template_count,
            flat_features.shape[-1],
        )
    )
