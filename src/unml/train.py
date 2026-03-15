from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPTokenizer

from .data import CIFAR10_CLASSES, build_loaders, build_text_inputs
from .evaluate import evaluate_classification
from .model import ModelConfig, LightweightVLM, save_checkpoint
from .tracker import log_finetune_epoch, log_finetune_summary
from .utils import format_metrics, get_device, save_json, set_seed, tensor_to_float


@dataclass
class FineTuneConfig:
    data_dir: str
    split_path: str
    output_dir: str
    model_name: str = "openai/clip-vit-base-patch32"
    prompt_template: str = "a photo of a {}"
    adapter_rank: int = 16
    adapter_alpha: float = 16.0
    train_logit_scale: bool = True
    batch_size: int = 128
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 5
    max_train_steps: int = -1
    seed: int = 42
    device: str = "auto"


def _evaluate_all(model, loaders, class_text_inputs, device: torch.device) -> Dict[str, float]:
    retain_val = evaluate_classification(model, loaders["retain_val"], class_text_inputs, device)
    test_all = evaluate_classification(model, loaders["test_all"], class_text_inputs, device)
    test_retain = evaluate_classification(model, loaders["test_retain"], class_text_inputs, device)
    forget_train = evaluate_classification(model, loaders["forget"], class_text_inputs, device)
    return {
        "retain_val_acc": retain_val["accuracy"],
        "retain_val_loss": retain_val["loss"],
        "test_all_acc": test_all["accuracy"],
        "test_retain_acc": test_retain["accuracy"],
        "forget_train_acc": forget_train["accuracy"],
    }


def run_finetuning(cfg: FineTuneConfig) -> Dict[str, str | float]:
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
    loaders = build_loaders(
        data_dir=cfg.data_dir,
        split_path=cfg.split_path,
        image_processor=image_processor,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model_cfg = ModelConfig(
        model_name=cfg.model_name,
        adapter_rank=cfg.adapter_rank,
        adapter_alpha=cfg.adapter_alpha,
        train_logit_scale=cfg.train_logit_scale,
    )

    model = LightweightVLM.from_config(model_cfg).to(device)

    # Frozen CLIP backbone in eval mode keeps deterministic behavior.
    model.clip.eval() # Evaluation mode: Disables certain layers like Dropout and Batch Normalization that behave differently during training v/s inference

    save_checkpoint(
        str(ckpt_dir / "base_init.pt"),
        model,
        extra={"stage": "base_init", "note": "randomly initialized adapters"},
    )

    optimizer = AdamW(model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(1, len(loaders["finetune_train"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    best_metric = -1.0
    best_path = ckpt_dir / "finetuned_best.pt"
    global_step = 0
    show_progress = os.environ.get("UNML_TQDM", "0") == "1"

    for epoch in range(cfg.epochs):
        model.train()
        model.clip.eval()
        epoch_losses = []
        progress = tqdm(
            loaders["finetune_train"],
            desc=f"finetune epoch {epoch+1}/{cfg.epochs}",
            disable=not show_progress,
        )
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model.class_logits(
                pixel_values=batch["pixel_values"],
                class_input_ids=class_text_inputs["input_ids"].to(device),
                class_attention_mask=class_text_inputs["attention_mask"].to(device),
            )
            loss = F.cross_entropy(logits, batch["labels"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_losses.append(tensor_to_float(loss))
            if show_progress:
                progress.set_postfix(loss=f"{tensor_to_float(loss):.4f}", step=global_step)
            elif global_step % 25 == 0:
                print(
                    f"[finetune] epoch={epoch + 1} step={global_step} loss={tensor_to_float(loss):.4f}",
                    flush=True,
                )

            if cfg.max_train_steps > 0 and global_step >= cfg.max_train_steps:
                break

        eval_metrics = _evaluate_all(model, loaders, class_text_inputs, device)
        epoch_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        full_metrics = {"epoch": float(epoch + 1), "train_loss": epoch_loss, **eval_metrics}
        print(f"[finetune] {format_metrics(full_metrics)}", flush=True)

        log_finetune_epoch(cfg.__dict__, epoch + 1, epoch_loss, eval_metrics)

        if eval_metrics["retain_val_acc"] > best_metric:
            best_metric = eval_metrics["retain_val_acc"]
            save_checkpoint(
                str(best_path),
                model,
                extra={"stage": "finetuned", "epoch": epoch + 1, "metrics": eval_metrics},
            )

        if cfg.max_train_steps > 0 and global_step >= cfg.max_train_steps:
            break

    final_path = ckpt_dir / "finetuned_last.pt"
    save_checkpoint(
        str(final_path),
        model,
        extra={"stage": "finetuned_last", "best_retain_val_acc": best_metric},
    )

    best_model_metrics = _evaluate_all(model, loaders, class_text_inputs, device)
    metrics_path = metrics_dir / "finetune_metrics.json"
    save_json(
        {
            "best_retain_val_acc": best_metric,
            "final_metrics": best_model_metrics,
            "global_steps": global_step,
            "class_names": CIFAR10_CLASSES,
        },
        metrics_path,
    )

    log_finetune_summary(cfg.__dict__, best_metric, cfg.epochs)

    return {
        "base_checkpoint": str(ckpt_dir / "base_init.pt"),
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "metrics_path": str(metrics_path),
        "best_retain_val_acc": best_metric,
    }
