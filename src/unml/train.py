from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPTokenizer

from .data import build_loaders, build_text_inputs, load_split_metadata
from .evaluate import build_class_text_features, evaluate_classification
from .model import ModelConfig, LightweightVLM, precision_context, save_checkpoint
from helpers.tracker import log_finetune_epoch, log_finetune_summary
from .utils import format_metrics, get_device, save_json, set_seed, tensor_to_float


@dataclass
class FineTuneConfig:
    data_dir: str
    split_path: str
    output_dir: str
    dataset_name: str = "cifar10"
    model_name: str = "openai/clip-vit-base-patch32"
    prompt_template: str = "a photo of a {}"
    adapter_rank: int = 8
    adapter_alpha: float = 8.0
    adapter_type: str = "post_projection"
    lora_rank: int = 8
    lora_alpha: float = 8.0
    lora_layers: str | list[int] = "all"
    lora_targets: tuple[str, ...] | list[str] = ("q_proj", "v_proj")
    train_logit_scale: bool = True
    precision: str = "fp32"
    gradient_checkpointing: bool = False
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 5
    max_train_steps: int = -1
    seed: int = 42
    device: str = "auto"


def _evaluate_all(model, loaders, class_text_inputs, device: torch.device) -> Dict[str, float]:
    model.eval()
    class_text_features = build_class_text_features(
        model, class_text_inputs, device
    )
    evaluation_args = {
        "class_text_inputs": class_text_inputs,
        "device": device,
        "class_text_features": class_text_features,
    }
    retain_val = evaluate_classification(
        model, loaders["retain_val"], **evaluation_args
    )
    test_all = evaluate_classification(
        model, loaders["test_all"], **evaluation_args
    )
    test_retain = evaluate_classification(
        model, loaders["test_retain"], **evaluation_args
    )
    forget_train = evaluate_classification(
        model, loaders["forget"], **evaluation_args
    )
    return {
        "retain_val_acc": retain_val["accuracy"],
        "retain_val_loss": retain_val["loss"],
        "test_all_acc": test_all["accuracy"],
        "test_retain_acc": test_retain["accuracy"],
        "forget_train_acc": forget_train["accuracy"],
    }


def run_finetuning(cfg: FineTuneConfig) -> Dict[str, str | float]:
    if cfg.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    output_dir = Path(cfg.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    image_processor = CLIPImageProcessor.from_pretrained(cfg.model_name)
    tokenizer = CLIPTokenizer.from_pretrained(cfg.model_name)
    _, dataset_spec, class_names = load_split_metadata(
        cfg.split_path, cfg.dataset_name
    )
    class_text_inputs = build_text_inputs(
        tokenizer,
        class_names=class_names,
        template=cfg.prompt_template,
    )
    loaders = build_loaders(
        data_dir=cfg.data_dir,
        split_path=cfg.split_path,
        image_processor=image_processor,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        dataset_name=dataset_spec.name,
    )

    model_cfg = ModelConfig(
        model_name=cfg.model_name,
        adapter_rank=cfg.adapter_rank,
        adapter_alpha=cfg.adapter_alpha,
        adapter_type=cfg.adapter_type,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_layers=cfg.lora_layers,
        lora_targets=cfg.lora_targets,
        train_logit_scale=cfg.train_logit_scale,
        precision=cfg.precision,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    model = LightweightVLM.from_config(model_cfg).to(device)

    model.clip.eval()
    class_text_inputs = {
        key: value.to(device) for key, value in class_text_inputs.items()
    }
    cached_train_text_features = None
    if model.can_cache_text_features_during_training():
        model.eval()
        cached_train_text_features = build_class_text_features(
            model, class_text_inputs, device
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    save_checkpoint(
        str(ckpt_dir / "base_init.pt"),
        model,
        extra={
            "stage": "base_init",
            "note": "randomly initialized adapters",
            "dataset": dataset_spec.name,
            "class_names": class_names,
            "architecture": model.architecture_summary(),
        },
    )

    optimizer = AdamW(model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    updates_per_epoch = math.ceil(
        len(loaders["finetune_train"]) / cfg.gradient_accumulation_steps
    )
    total_steps = cfg.epochs * max(1, updates_per_epoch)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    best_metric = -1.0
    best_path = ckpt_dir / "finetuned_best.pt"
    global_step = 0
    processed_examples = 0
    training_started = time.perf_counter()
    show_progress = os.environ.get("UNML_TQDM", "0") == "1"
    final_metrics: Dict[str, float] | None = None

    for epoch in range(cfg.epochs):
        model.set_train_mode()
        epoch_losses = []
        progress = tqdm(
            loaders["finetune_train"],
            desc=f"finetune epoch {epoch+1}/{cfg.epochs}",
            disable=not show_progress,
        )
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(progress):
            batch = {k: v.to(device) for k, v in batch.items()}
            processed_examples += int(batch["labels"].numel())
            with precision_context(cfg.precision, device):
                if cached_train_text_features is None:
                    logits = model.class_logits(
                        pixel_values=batch["pixel_values"],
                        class_input_ids=class_text_inputs["input_ids"],
                        class_attention_mask=class_text_inputs["attention_mask"],
                    )
                else:
                    logits = model.class_logits_from_text_features(
                        pixel_values=batch["pixel_values"],
                        class_text_features=cached_train_text_features,
                    )
                loss = F.cross_entropy(logits, batch["labels"])

            (loss / cfg.gradient_accumulation_steps).backward()
            should_step = (
                (batch_index + 1) % cfg.gradient_accumulation_steps == 0
                or batch_index + 1 == len(loaders["finetune_train"])
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    list(model.trainable_parameters()), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            epoch_losses.append(tensor_to_float(loss))
            if show_progress:
                progress.set_postfix(loss=f"{tensor_to_float(loss):.4f}", step=global_step)
            elif should_step and global_step % 25 == 0:
                print(
                    f"[finetune] epoch={epoch + 1} step={global_step} loss={tensor_to_float(loss):.4f}",
                    flush=True,
                )

            if (
                should_step
                and cfg.max_train_steps > 0
                and global_step >= cfg.max_train_steps
            ):
                break

        eval_metrics = _evaluate_all(model, loaders, class_text_inputs, device)
        final_metrics = eval_metrics
        epoch_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        full_metrics = {"epoch": float(epoch + 1), "train_loss": epoch_loss, **eval_metrics}
        print(f"[finetune] {format_metrics(full_metrics)}", flush=True)

        log_finetune_epoch(cfg.__dict__, epoch + 1, epoch_loss, eval_metrics)

        if eval_metrics["retain_val_acc"] > best_metric:
            best_metric = eval_metrics["retain_val_acc"]
            save_checkpoint(
                str(best_path),
                model,
                extra={
                    "stage": "finetuned",
                    "epoch": epoch + 1,
                    "dataset": dataset_spec.name,
                    "class_names": class_names,
                    "architecture": model.architecture_summary(),
                    "metrics": eval_metrics,
                },
            )

        if cfg.max_train_steps > 0 and global_step >= cfg.max_train_steps:
            break

    training_seconds = time.perf_counter() - training_started
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    benchmark = {
        "training_seconds": training_seconds,
        "processed_examples": processed_examples,
        "examples_per_second": (
            processed_examples / training_seconds if training_seconds > 0 else 0.0
        ),
        "peak_gpu_memory_mb": peak_memory_mb,
        "physical_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "effective_batch_size": (
            cfg.batch_size * cfg.gradient_accumulation_steps
        ),
        "device": str(device),
        "precision": cfg.precision,
    }
    if final_metrics is None:
        raise RuntimeError("Fine-tuning completed without an evaluation snapshot")
    metrics_path = metrics_dir / "finetune_metrics.json"
    save_json(
        {
            "best_retain_val_acc": best_metric,
            "final_metrics": final_metrics,
            "global_steps": global_step,
            "dataset": dataset_spec.name,
            "class_names": class_names,
            "architecture": model.architecture_summary(),
            "benchmark": benchmark,
            "cached_train_text_features": (
                cached_train_text_features is not None
            ),
        },
        metrics_path,
    )

    log_finetune_summary(cfg.__dict__, best_metric, cfg.epochs)

    return {
        "base_checkpoint": str(ckpt_dir / "base_init.pt"),
        "best_checkpoint": str(best_path),
        "metrics_path": str(metrics_path),
        "best_retain_val_acc": best_metric,
    }
