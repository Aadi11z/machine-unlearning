from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import trange
from transformers import CLIPImageProcessor, CLIPTokenizer

from .data import CIFAR10_CLASSES, build_loaders, build_text_inputs, cycle_loader
from .evaluate import evaluate_classification
from .model import load_checkpoint, save_checkpoint
from .utils import format_metrics, get_device, save_json, set_seed, tensor_to_float


UNLEARNING_METHODS = {
    "retain_only",
    "ga_kl",
    "counterfactual_rebind",
}


@dataclass
class UnlearnConfig:
    data_dir: str
    split_path: str
    finetuned_checkpoint: str
    output_dir: str
    method: str
    model_name: str = "openai/clip-vit-base-patch32"
    prompt_template: str = "a photo of a {}"
    batch_size: int = 128
    num_workers: int = 4
    steps: int = 500
    lr: float = 5e-4
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "auto"
    kl_temperature: float = 1.5
    kl_weight: float = 1.0
    ga_weight: float = 1.0
    cf_weight: float = 1.0
    margin_weight: float = 0.5
    margin: float = 0.2


def _sample_counterfactual(labels: torch.Tensor, num_classes: int, rng: random.Random) -> torch.Tensor:
    y_cf = labels.clone()
    for i in range(labels.size(0)):
        y = int(labels[i].item())
        candidates = [c for c in range(num_classes) if c != y]
        y_cf[i] = int(rng.choice(candidates))
    return y_cf


def _kl_div(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature**2)


def _eval_snapshot(model, loaders, class_text_inputs, device: torch.device) -> Dict[str, float]:
    retain_train = evaluate_classification(model, loaders["retain_train"], class_text_inputs, device)
    retain_val = evaluate_classification(model, loaders["retain_val"], class_text_inputs, device)
    forget = evaluate_classification(model, loaders["forget"], class_text_inputs, device)
    test_retain = evaluate_classification(model, loaders["test_retain"], class_text_inputs, device)
    test_all = evaluate_classification(model, loaders["test_all"], class_text_inputs, device)
    return {
        "retain_train_acc": retain_train["accuracy"],
        "retain_val_acc": retain_val["accuracy"],
        "forget_acc": forget["accuracy"],
        "test_retain_acc": test_retain["accuracy"],
        "test_all_acc": test_all["accuracy"],
    }


def run_unlearning(cfg: UnlearnConfig) -> Dict[str, str | float]:
    if cfg.method not in UNLEARNING_METHODS:
        raise ValueError(f"Unsupported method={cfg.method}. Choose from {sorted(UNLEARNING_METHODS)}")

    set_seed(cfg.seed)
    device = get_device(cfg.device)

    output_dir = Path(cfg.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    image_processor = CLIPImageProcessor.from_pretrained(cfg.model_name)
    tokenizer = CLIPTokenizer.from_pretrained(cfg.model_name)
    class_text_inputs = build_text_inputs(
        tokenizer,
        class_names=CIFAR10_CLASSES,
        template=cfg.prompt_template,
    )
    class_text_inputs = {k: v.to(device) for k, v in class_text_inputs.items()}

    loaders = build_loaders(
        data_dir=cfg.data_dir,
        split_path=cfg.split_path,
        image_processor=image_processor,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model, finetune_meta = load_checkpoint(cfg.finetuned_checkpoint, map_location=device)
    model = model.to(device)
    model.clip.eval()

    teacher = copy.deepcopy(model).eval().to(device)
    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = AdamW(model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    forget_iter = cycle_loader(loaders["forget"])
    retain_iter = cycle_loader(loaders["retain_train"])
    rng = random.Random(cfg.seed)

    losses = []
    show_progress = os.environ.get("UNML_TQDM", "0") == "1"

    for step in trange(cfg.steps, desc=f"unlearn:{cfg.method}", disable=not show_progress):
        model.train()
        model.clip.eval()

        batch_r = next(retain_iter)
        batch_r = {k: v.to(device) for k, v in batch_r.items()}
        logits_r = model.class_logits(
            pixel_values=batch_r["pixel_values"],
            class_input_ids=class_text_inputs["input_ids"],
            class_attention_mask=class_text_inputs["attention_mask"],
        )
        with torch.no_grad():
            t_logits_r = teacher.class_logits(
                pixel_values=batch_r["pixel_values"],
                class_input_ids=class_text_inputs["input_ids"],
                class_attention_mask=class_text_inputs["attention_mask"],
            )

        retain_kl = _kl_div(logits_r, t_logits_r, cfg.kl_temperature)

        if cfg.method == "retain_only":
            loss = F.cross_entropy(logits_r, batch_r["labels"])
        else:
            batch_f = next(forget_iter)
            batch_f = {k: v.to(device) for k, v in batch_f.items()}
            logits_f = model.class_logits(
                pixel_values=batch_f["pixel_values"],
                class_input_ids=class_text_inputs["input_ids"],
                class_attention_mask=class_text_inputs["attention_mask"],
            )

            if cfg.method == "ga_kl":
                forget_term = -F.cross_entropy(logits_f, batch_f["labels"])
                loss = cfg.ga_weight * forget_term + cfg.kl_weight * retain_kl

            elif cfg.method == "counterfactual_rebind":
                y_true = batch_f["labels"]
                y_cf = _sample_counterfactual(y_true, len(CIFAR10_CLASSES), rng).to(device)
                rebind_loss = F.cross_entropy(logits_f, y_cf)
                true_logits = logits_f.gather(1, y_true.unsqueeze(1)).squeeze(1)
                cf_logits = logits_f.gather(1, y_cf.unsqueeze(1)).squeeze(1)
                margin_loss = F.relu(true_logits - cf_logits + cfg.margin).mean()

                loss = (
                    cfg.cf_weight * rebind_loss
                    + cfg.margin_weight * margin_loss
                    + cfg.kl_weight * retain_kl
                )
            else:
                raise RuntimeError(f"Unreachable method: {cfg.method}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), max_norm=1.0)
        optimizer.step()

        losses.append(tensor_to_float(loss))

        if (step + 1) % max(10, cfg.steps // 5) == 0:
            recent_loss = sum(losses[-10:]) / min(10, len(losses))
            print(f"[unlearn:{cfg.method}] step={step+1} loss={recent_loss:.4f}", flush=True)

    model.eval()
    summary_metrics = _eval_snapshot(model, loaders, class_text_inputs, device)
    summary_metrics["avg_train_loss"] = float(sum(losses) / max(1, len(losses)))
    summary_metrics["steps"] = float(cfg.steps)
    print(f"[unlearn:{cfg.method}] {format_metrics(summary_metrics)}", flush=True)

    ckpt_path = ckpt_dir / f"unlearn_{cfg.method}.pt"
    save_checkpoint(
        str(ckpt_path),
        model,
        extra={
            "stage": "unlearned",
            "method": cfg.method,
            "finetuned_checkpoint": cfg.finetuned_checkpoint,
            "finetune_meta": finetune_meta,
            "metrics": summary_metrics,
        },
    )

    metrics_path = metrics_dir / f"unlearn_{cfg.method}_metrics.json"
    save_json(
        {
            "method": cfg.method,
            "summary_metrics": summary_metrics,
            "config": cfg.__dict__,
        },
        metrics_path,
    )

    return {
        "checkpoint": str(ckpt_path),
        "metrics_path": str(metrics_path),
        **summary_metrics,
    }
