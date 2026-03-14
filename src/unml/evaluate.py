from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import move_to_device, tensor_to_float


@torch.no_grad() # needed so that Pytorch doesnt build autograd graphs during execution
def evaluate_classification(model, loader: DataLoader, class_text_inputs: Dict[str, torch.Tensor], device: torch.device, max_batches: int | None = None) -> Dict[str, float]:
    # Performs supervised eval
    model.eval()
    total = 0
    correct = 0
    losses = []
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches: 
            # if you want early stopping, declare max_batches
            break
        batch = move_to_device(batch, device)
        logits = model.class_logits(
            pixel_values=batch["pixel_values"],
            class_input_ids=class_text_inputs["input_ids"].to(device),
            class_attention_mask=class_text_inputs["attention_mask"].to(device),
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
def collect_true_class_confidences(model, loader: DataLoader, class_text_inputs: Dict[str, torch.Tensor], device: torch.device, max_samples: int | None = None):
    model.eval()
    scores = []
    labels_all = []
    indices_all = []
    for batch in loader:
        batch = move_to_device(batch, device)
        logits = model.class_logits(
            pixel_values=batch["pixel_values"],
            class_input_ids=class_text_inputs["input_ids"].to(device),
            class_attention_mask=class_text_inputs["attention_mask"].to(device),
        )
        probs = logits.softmax(dim=-1)
        labels = batch["labels"]
        true_scores = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        scores.append(true_scores.detach().cpu())
        labels_all.append(labels.detach().cpu())
        indices_all.append(batch["indices"].detach().cpu())

        if max_samples is not None and sum(t.shape[0] for t in scores) >= max_samples:
            break

    score_tensor = torch.cat(scores, dim=0)
    label_tensor = torch.cat(labels_all, dim=0)
    index_tensor = torch.cat(indices_all, dim=0)
    if max_samples is not None:
        score_tensor = score_tensor[:max_samples]
        label_tensor = label_tensor[:max_samples]
        index_tensor = index_tensor[:max_samples]

    return {
        "scores": np.asarray(score_tensor.tolist(), dtype=np.float32),
        "labels": np.asarray(label_tensor.tolist(), dtype=np.int64),
        "indices": np.asarray(index_tensor.tolist(), dtype=np.int64),
    }
