from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor, CLIPTokenizer

from .data import (
    CLIPCollator,
    build_canonical_prompt_inputs,
    load_dataset_pair,
    load_split_metadata,
    validate_checkpoint_dataset,
)
from .evaluate import build_class_text_features
from .model import load_checkpoint, precision_context
from .utils import ensure_dir, get_device, transformers_offline


@dataclass
class ImageCheckpointPredictor:
    """A loaded checkpoint and its fixed CIFAR class prompt features."""

    model: torch.nn.Module
    metadata: Mapping[str, Any]
    class_names: Sequence[str]
    image_processor: CLIPImageProcessor
    class_text_features: torch.Tensor
    device: torch.device


def _class_name_to_id(class_names: Sequence[str], class_name: str) -> int:
    normalized = class_name.strip().lower().replace("_", " ")
    matches = {
        name.strip().lower().replace("_", " "): index
        for index, name in enumerate(class_names)
    }
    if normalized not in matches:
        raise ValueError(f"Unknown class name={class_name!r}")
    return matches[normalized]


@torch.no_grad()
def _probabilities_from_pixels(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    class_text_features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with precision_context(model.cfg.precision, device):
        logits = model.class_logits_from_text_features(
            pixel_values=pixel_values.to(device),
            class_text_features=class_text_features,
        )
    return logits.float().softmax(dim=-1)


def load_image_checkpoint_predictor(
    *,
    checkpoint_path: str,
    dataset_name: str,
    class_names: Sequence[str],
    prompt_template: str,
    device_name: str,
    expected_model_name: str | None = None,
    local_files_only: bool = False,
) -> ImageCheckpointPredictor:
    """Load one immutable checkpoint for label-free image probing."""
    device = get_device(device_name)
    model, metadata = load_checkpoint(checkpoint_path, map_location=device)
    validate_checkpoint_dataset(metadata, dataset_name, checkpoint_path)
    if expected_model_name is not None and model.cfg.model_name != expected_model_name:
        raise ValueError(
            f"Checkpoint {checkpoint_path} uses {model.cfg.model_name}, "
            f"expected {expected_model_name}"
        )

    model = model.to(device).eval()
    model_name = model.cfg.model_name
    image_processor = CLIPImageProcessor.from_pretrained(
        model_name,
        local_files_only=local_files_only or transformers_offline(),
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only or transformers_offline(),
    )
    class_text_inputs = build_canonical_prompt_inputs(
        tokenizer,
        class_names=class_names,
        dataset_name=dataset_name,
    )
    class_text_features = build_class_text_features(
        model, class_text_inputs, device
    )
    return ImageCheckpointPredictor(
        model=model,
        metadata=metadata,
        class_names=tuple(class_names),
        image_processor=image_processor,
        class_text_features=class_text_features,
        device=device,
    )


@torch.no_grad()
def predict_probe_image(
    predictor: ImageCheckpointPredictor,
    image: Any,
    *,
    target_class_name: str,
    top_k: int,
) -> dict[str, Any]:
    """Return CIFAR class rankings for an image without assuming a true label."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not hasattr(image, "convert"):
        raise ValueError("Probe image must be a decoded image object")

    image = image.convert("RGB")
    inputs = predictor.image_processor(images=image, return_tensors="pt")
    probabilities = _probabilities_from_pixels(
        predictor.model,
        inputs["pixel_values"],
        predictor.class_text_features,
        predictor.device,
    )[0]
    count = min(top_k, len(predictor.class_names))
    values, class_ids = probabilities.topk(count)
    target_class_id = _class_name_to_id(predictor.class_names, target_class_name)
    target_probability = float(probabilities[target_class_id].item())
    target_rank = int((probabilities > target_probability).sum().item()) + 1
    all_scores = probabilities.detach().cpu().tolist()

    return {
        "top_k": [
            {
                "rank": rank,
                "class_id": int(class_id.item()),
                "class_name": predictor.class_names[int(class_id.item())],
                "probability": float(value.item()),
            }
            for rank, (value, class_id) in enumerate(
                zip(values.detach().cpu(), class_ids.detach().cpu()), start=1
            )
        ],
        "target_class_id": target_class_id,
        "target_class_name": predictor.class_names[target_class_id],
        "target_probability": target_probability,
        "target_rank": target_rank,
        "class_scores": {
            class_name: float(probability)
            for class_name, probability in zip(predictor.class_names, all_scores)
        },
    }


def resolve_probe_classes(
    *,
    class_ids: Sequence[int],
    class_names_requested: Sequence[str],
    class_names: Sequence[str],
    target_classes: Sequence[int],
) -> list[int]:
    resolved = {int(class_id) for class_id in class_ids}
    normalized_names = {
        name.strip().lower().replace("_", " "): index
        for index, name in enumerate(class_names)
    }
    for requested in class_names_requested:
        normalized = requested.strip().lower().replace("_", " ")
        if normalized not in normalized_names:
            raise ValueError(
                f"Unknown class name={requested!r}; choose from {list(class_names)}"
            )
        resolved.add(normalized_names[normalized])
    if not resolved:
        resolved.update(int(class_id) for class_id in target_classes)
    invalid = sorted(
        class_id
        for class_id in resolved
        if class_id < 0 or class_id >= len(class_names)
    )
    if invalid:
        raise ValueError(
            f"Class ids outside [0, {len(class_names) - 1}]: {invalid}"
        )
    if not resolved:
        raise ValueError("No probe indices or classes were selected")
    return sorted(resolved)


def select_probe_indices(
    *,
    dataset: Dataset,
    explicit_indices: Sequence[int],
    class_ids: Sequence[int],
    limit_per_class: int,
) -> list[int]:
    if limit_per_class < 1:
        raise ValueError("limit_per_class must be at least 1")
    dataset_size = len(dataset)
    selected = []
    seen = set()
    for index in explicit_indices:
        index = int(index)
        if index < 0 or index >= dataset_size:
            raise ValueError(
                f"Dataset index {index} outside [0, {dataset_size - 1}]"
            )
        if index not in seen:
            selected.append(index)
            seen.add(index)

    targets = getattr(dataset, "targets", None)
    if targets is None and class_ids:
        raise ValueError("Dataset does not expose targets for class probing")
    for class_id in class_ids:
        count = 0
        for index, label in enumerate(targets):
            if int(label) != int(class_id) or index in seen:
                continue
            selected.append(index)
            seen.add(index)
            count += 1
            if count >= limit_per_class:
                break
    if not selected:
        raise ValueError("Probe selection produced no samples")
    return selected


def default_probe_indices(
    *,
    source: str,
    split: Mapping[str, Any],
    dataset: Dataset,
    limit_per_class: int,
) -> list[int]:
    if source != "train":
        return []
    targets = getattr(dataset, "targets", None)
    forget_indices = [int(index) for index in split.get("forget_indices", [])]
    if targets is None:
        return forget_indices[:limit_per_class]
    counts: dict[int, int] = {}
    selected = []
    for index in forget_indices:
        class_id = int(targets[index])
        if counts.get(class_id, 0) >= limit_per_class:
            continue
        selected.append(index)
        counts[class_id] = counts.get(class_id, 0) + 1
    return selected


def hierarchy_group(class_id: int, split: Mapping[str, Any]) -> str:
    if class_id in set(split.get("target_classes", split.get("forget_classes", []))):
        return "target"
    if class_id in set(split.get("sibling_classes", [])):
        return "sibling"
    if class_id in set(split.get("unrelated_classes", [])):
        return "unrelated"
    return "retain"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"


def _prepare_batch(
    dataset: Dataset,
    indices: Sequence[int],
    image_processor: CLIPImageProcessor,
) -> tuple[dict[str, torch.Tensor], list[Any]]:
    samples = []
    images = []
    for index in indices:
        image, label = dataset[int(index)]
        images.append(image.copy())
        samples.append((image, int(label), int(index)))
    return CLIPCollator(image_processor)(samples), images


@torch.no_grad()
def _predict_checkpoint(
    checkpoint_path: str,
    dataset_name: str,
    batch: dict[str, torch.Tensor],
    class_text_inputs,
    device: torch.device,
    top_k: int,
    expected_model_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, metadata = load_checkpoint(checkpoint_path, map_location=device)
    validate_checkpoint_dataset(metadata, dataset_name, checkpoint_path)
    if model.cfg.model_name != expected_model_name:
        raise ValueError(
            f"Checkpoint {checkpoint_path} uses {model.cfg.model_name}, "
            f"expected {expected_model_name}"
        )
    model = model.to(device).eval()
    text_features = build_class_text_features(
        model, class_text_inputs, device
    )
    probabilities = _probabilities_from_pixels(
        model,
        batch["pixel_values"],
        text_features,
        device,
    )
    labels = batch["labels"].to(device)
    true_confidences = probabilities.gather(
        1, labels.unsqueeze(1)
    ).squeeze(1)
    values, predictions = probabilities.topk(
        min(top_k, probabilities.shape[1]), dim=-1
    )
    rows = []
    for row_index in range(probabilities.shape[0]):
        rows.append(
            {
                "predicted_class_id": int(predictions[row_index, 0].item()),
                "prediction_confidence": float(values[row_index, 0].item()),
                "true_confidence": float(true_confidences[row_index].item()),
                "top_k_class_ids": [
                    int(value)
                    for value in predictions[row_index].detach().cpu().tolist()
                ],
                "top_k_confidences": [
                    float(value)
                    for value in values[row_index].detach().cpu().tolist()
                ],
            }
        )
    return metadata, rows


def run_checkpoint_probe(
    *,
    data_dir: str,
    split_path: str,
    dataset_name: str,
    source: str,
    indices: Sequence[int],
    class_ids: Sequence[int],
    class_names_requested: Sequence[str],
    limit_per_class: int,
    reference_checkpoint: str,
    candidate_names: Sequence[str],
    candidate_checkpoints: Sequence[str],
    output_dir: str,
    prompt_template: str,
    top_k: int,
    device_name: str,
    save_images: bool,
    local_files_only: bool = False,
) -> dict[str, Any]:
    import pandas as pd

    if source not in {"train", "test"}:
        raise ValueError("source must be train or test")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if len(candidate_names) != len(candidate_checkpoints):
        raise ValueError("Candidate names and checkpoints must have equal length")
    if not candidate_names:
        raise ValueError("At least one candidate checkpoint is required")
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("Candidate names must be unique")

    split, spec, class_names = load_split_metadata(split_path, dataset_name)
    train_dataset, test_dataset = load_dataset_pair(data_dir, spec.name)
    dataset = train_dataset if source == "train" else test_dataset
    explicit_indices = list(indices)
    if not indices and not class_ids and not class_names_requested:
        explicit_indices = default_probe_indices(
            source=source,
            split=split,
            dataset=dataset,
            limit_per_class=limit_per_class,
        )
    probe_classes = (
        []
        if explicit_indices and not class_ids and not class_names_requested
        else resolve_probe_classes(
            class_ids=class_ids,
            class_names_requested=class_names_requested,
            class_names=class_names,
            target_classes=split.get(
                "target_classes", split.get("forget_classes", [])
            ),
        )
    )
    selected_indices = select_probe_indices(
        dataset=dataset,
        explicit_indices=explicit_indices,
        class_ids=probe_classes,
        limit_per_class=limit_per_class,
    )

    reference_payload = torch.load(reference_checkpoint, map_location="cpu", weights_only=False)
    reference_model_name = reference_payload["model_config"]["model_name"]
    image_processor = CLIPImageProcessor.from_pretrained(
        reference_model_name,
        local_files_only=local_files_only or transformers_offline(),
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        reference_model_name,
        local_files_only=local_files_only or transformers_offline(),
    )
    class_text_inputs = build_canonical_prompt_inputs(
        tokenizer,
        class_names=class_names,
        dataset_name=dataset_name,
    )
    batch, images = _prepare_batch(dataset, selected_indices, image_processor)
    device = get_device(device_name)

    checkpoint_entries = [
        ("reference", reference_checkpoint),
        *zip(candidate_names, candidate_checkpoints),
    ]
    predictions_by_model = {}
    metadata_by_model = {}
    for name, checkpoint in checkpoint_entries:
        metadata, predictions = _predict_checkpoint(
            checkpoint,
            spec.name,
            batch,
            class_text_inputs,
            device,
            top_k,
            reference_model_name,
        )
        predictions_by_model[name] = predictions
        metadata_by_model[name] = metadata

    reference_predictions = predictions_by_model["reference"]
    rows = []
    for sample_position, dataset_index in enumerate(selected_indices):
        true_class_id = int(batch["labels"][sample_position].item())
        for model_name, _ in checkpoint_entries:
            prediction = predictions_by_model[model_name][sample_position]
            predicted_id = prediction["predicted_class_id"]
            top_ids = prediction["top_k_class_ids"]
            rows.append(
                {
                    "source": source,
                    "dataset_index": int(dataset_index),
                    "true_class_id": true_class_id,
                    "true_class_name": class_names[true_class_id],
                    "hierarchy_group": hierarchy_group(true_class_id, split),
                    "model": model_name,
                    "predicted_class_id": predicted_id,
                    "predicted_class_name": class_names[predicted_id],
                    "correct": predicted_id == true_class_id,
                    "true_confidence": prediction["true_confidence"],
                    "reference_true_confidence": reference_predictions[
                        sample_position
                    ]["true_confidence"],
                    "true_confidence_delta": (
                        prediction["true_confidence"]
                        - reference_predictions[sample_position][
                            "true_confidence"
                        ]
                    ),
                    "prediction_confidence": prediction[
                        "prediction_confidence"
                    ],
                    "top_k_class_ids": json.dumps(top_ids),
                    "top_k_class_names": json.dumps(
                        [class_names[class_id] for class_id in top_ids]
                    ),
                    "top_k_confidences": json.dumps(
                        prediction["top_k_confidences"]
                    ),
                }
            )

    output_path = Path(output_dir)
    ensure_dir(output_path)
    csv_path = output_path / "probe.csv"
    json_path = output_path / "probe.json"
    markdown_path = output_path / "probe.md"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": spec.name,
                "request": split.get("request_name"),
                "source": source,
                "indices": selected_indices,
                "probe_classes": probe_classes,
                "class_names": class_names,
                "reference_checkpoint": reference_checkpoint,
                "candidates": dict(zip(candidate_names, candidate_checkpoints)),
                "metadata": metadata_by_model,
                "rows": rows,
            },
            handle,
            indent=2,
            default=str,
        )
    compact = pd.DataFrame(rows)[
        [
            "dataset_index",
            "true_class_name",
            "hierarchy_group",
            "model",
            "predicted_class_name",
            "correct",
            "true_confidence",
            "true_confidence_delta",
        ]
    ]
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# Checkpoint Probe\n\n")
        handle.write(compact.to_markdown(index=False))
        handle.write("\n")

    image_paths = []
    if save_images:
        images_dir = output_path / "images"
        ensure_dir(images_dir)
        for dataset_index, image, label in zip(
            selected_indices, images, batch["labels"].tolist()
        ):
            image_path = images_dir / (
                f"{source}_{dataset_index}_{_safe_name(class_names[label])}.png"
            )
            image.save(image_path)
            image_paths.append(str(image_path))

    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "image_paths": image_paths,
        "sample_count": len(selected_indices),
        "model_count": len(checkpoint_entries),
    }
