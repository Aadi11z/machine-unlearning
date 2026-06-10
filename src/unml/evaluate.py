from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import move_to_device, tensor_to_float


@torch.no_grad()
def build_class_text_features(
    model,
    class_text_inputs: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    return model.encode_text(
        class_text_inputs["input_ids"].to(device),
        class_text_inputs["attention_mask"].to(device),
    )


@torch.no_grad() # needed so that Pytorch doesnt build autograd graphs during execution
def evaluate_classification(
    model,
    loader: DataLoader,
    class_text_inputs: Dict[str, torch.Tensor],
    device: torch.device,
    max_batches: int | None = None,
    class_text_features: torch.Tensor | None = None,
    non_blocking: bool = False,
) -> Dict[str, float]:
    # Performs supervised eval
    model.eval()
    if class_text_features is None:
        class_text_features = build_class_text_features(
            model, class_text_inputs, device
        )
    total = 0
    correct = 0
    losses = []
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches: 
            # if you want early stopping, declare max_batches
            break
        batch = move_to_device(batch, device, non_blocking=non_blocking)
        logits = model.class_logits_from_text_features(
            pixel_values=batch["pixel_values"],
            class_text_features=class_text_features,
        )
        labels = batch["labels"]
        loss = F.cross_entropy(logits, labels) # Per-Batch loss computation
        preds = logits.argmax(dim=-1) # gets the indices of the maximum prediction, index of the max value across a row in the tensor, used to get predicted class
        correct += int((preds == labels).sum().item()) # no. of correctly predicted samples
        total += int(labels.numel()) # no. of processed samples
        losses.append(tensor_to_float(loss)) # scalar loss values 

    acc = 0.0 if total == 0 else correct / total
    loss_value = float(np.mean(losses)) if losses else 0.0
    return {
        "accuracy": acc,
        "loss": loss_value,
        "n": float(total),
    }


@torch.no_grad()
def collect_classification_outputs(
    model,
    loader: DataLoader,
    class_text_inputs: Dict[str, torch.Tensor],
    device: torch.device,
    max_samples: int | None = None,
    class_text_features: torch.Tensor | None = None,
    non_blocking: bool = False,
    include_probabilities: bool = False,
    include_embeddings: bool = False,
) -> Dict[str, np.ndarray | float]:
    model.eval()
    if class_text_features is None:
        class_text_features = build_class_text_features(
            model, class_text_inputs, device
        )
    labels_all = []
    predictions_all = []
    indices_all = []
    true_scores_all = []
    probabilities_all = []
    embeddings_all = []
    losses = []
    collected = 0
    for batch in loader:
        batch = move_to_device(batch, device, non_blocking=non_blocking)
        if include_probabilities or include_embeddings:
            image_features = model.encode_images(batch["pixel_values"])
            logits = model.logits_from_embeddings(
                image_features, class_text_features
            )
        else:
            image_features = None
            logits = model.class_logits_from_text_features(
                pixel_values=batch["pixel_values"],
                class_text_features=class_text_features,
            )
        labels = batch["labels"]
        probabilities = logits.softmax(dim=-1)
        labels_all.append(labels.detach().cpu())
        predictions_all.append(logits.argmax(dim=-1).detach().cpu())
        indices_all.append(batch["indices"].detach().cpu())
        true_scores_all.append(
            probabilities.gather(1, labels.unsqueeze(1)).squeeze(1).detach().cpu()
        )
        if include_probabilities:
            probabilities_all.append(probabilities.detach().cpu())
        if include_embeddings:
            embeddings_all.append(image_features.detach().cpu())
        losses.append(
            F.cross_entropy(logits, labels, reduction="none").detach().cpu()
        )
        collected += int(labels.numel())
        if max_samples is not None and collected >= max_samples:
            break

    if not labels_all:
        empty: Dict[str, np.ndarray | float] = {
            "labels": np.empty(0, dtype=np.int64),
            "predictions": np.empty(0, dtype=np.int64),
            "indices": np.empty(0, dtype=np.int64),
            "true_scores": np.empty(0, dtype=np.float32),
            "sample_losses": np.empty(0, dtype=np.float32),
            "loss": 0.0,
        }
        if include_probabilities:
            empty["probabilities"] = np.empty(
                (0, int(class_text_features.shape[0])), dtype=np.float32
            )
        if include_embeddings:
            empty["embeddings"] = np.empty(
                (0, int(class_text_features.shape[1])), dtype=np.float32
            )
        return empty

    labels = torch.cat(labels_all)
    predictions = torch.cat(predictions_all)
    indices = torch.cat(indices_all)
    true_scores = torch.cat(true_scores_all)
    sample_losses = torch.cat(losses)
    all_probabilities = (
        torch.cat(probabilities_all) if include_probabilities else None
    )
    all_embeddings = (
        torch.cat(embeddings_all) if include_embeddings else None
    )
    if max_samples is not None:
        labels = labels[:max_samples]
        predictions = predictions[:max_samples]
        indices = indices[:max_samples]
        true_scores = true_scores[:max_samples]
        sample_losses = sample_losses[:max_samples]
        if all_probabilities is not None:
            all_probabilities = all_probabilities[:max_samples]
        if all_embeddings is not None:
            all_embeddings = all_embeddings[:max_samples]
    result: Dict[str, np.ndarray | float] = {
        "labels": labels.numpy().astype(np.int64, copy=False),
        "predictions": predictions.numpy().astype(np.int64, copy=False),
        "indices": indices.numpy().astype(np.int64, copy=False),
        "true_scores": true_scores.numpy().astype(np.float32, copy=False),
        "sample_losses": sample_losses.numpy().astype(
            np.float32, copy=False
        ),
        "loss": float(sample_losses.mean().item()),
    }
    if all_probabilities is not None:
        result["probabilities"] = all_probabilities.numpy().astype(
            np.float32, copy=False
        )
    if all_embeddings is not None:
        result["embeddings"] = all_embeddings.numpy().astype(
            np.float32, copy=False
        )
    return result


def summarize_classification_group(
    outputs: Dict[str, np.ndarray | float],
    class_ids: Sequence[int],
    num_classes: int,
) -> Dict[str, Any]:
    labels = np.asarray(outputs["labels"], dtype=np.int64)
    predictions = np.asarray(outputs["predictions"], dtype=np.int64)
    selected_classes = sorted(set(int(class_id) for class_id in class_ids))
    mask = np.isin(labels, selected_classes)
    selected_labels = labels[mask]
    selected_predictions = predictions[mask]

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    if selected_labels.size:
        np.add.at(confusion, (selected_labels, selected_predictions), 1)

    per_class = []
    class_accuracies = []
    for class_id in selected_classes:
        class_mask = selected_labels == class_id
        count = int(class_mask.sum())
        correct = int(
            (selected_predictions[class_mask] == class_id).sum()
        )
        accuracy = correct / count if count else None
        if accuracy is not None:
            class_accuracies.append(accuracy)
        per_class.append(
            {
                "class_id": class_id,
                "n": count,
                "correct": correct,
                "accuracy": accuracy,
            }
        )

    total = int(selected_labels.size)
    correct = int((selected_labels == selected_predictions).sum())
    return {
        "n": total,
        "accuracy": correct / total if total else None,
        "macro_accuracy": (
            float(np.mean(class_accuracies)) if class_accuracies else None
        ),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }
