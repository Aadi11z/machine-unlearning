from __future__ import annotations

import os
import math
import time
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPTokenizer

from .data import (
    build_canonical_prompt_inputs,
    build_loaders,
    load_split_metadata,
    validate_checkpoint_dataset,
)
from .evaluate import build_class_text_features, evaluate_classification
from .model import (
    LightweightVLM,
    load_checkpoint,
    load_recovery_checkpoint,
    precision_context,
    save_checkpoint,
    save_recovery_checkpoint,
    update_checkpoint_extra,
)
from .tracker import log_finetune_epoch, log_finetune_summary
from .utils import (
    format_metrics,
    get_device,
    load_json,
    move_to_device,
    save_json,
    set_seed,
    tensor_to_float,
)
from .utils import collect_run_provenance


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
    lora_dropout: float = 0.0
    train_logit_scale: bool = True
    precision: str = "fp32"
    gradient_checkpointing: bool = False
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2
    non_blocking: bool = False
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 5
    max_train_steps: int = -1
    seed: int = 42
    device: str = "auto"
    local_files_only: bool = False
    max_eval_batches: int | None = None
    evaluate_test: bool = True
    smoke_mode: bool = False
    training_mode: str = "finetune"
    train_loader_key: str = "finetune_train"
    initial_checkpoint: str | None = None
    resume_checkpoint: str | None = None
    source_metrics_path: str | None = None
    target_optimizer_steps: int | None = None
    target_scheduler_steps: int | None = None


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_oracle_schedule(
    source_metrics: Mapping[str, Any],
) -> tuple[int, int]:
    if bool(source_metrics.get("smoke_mode")):
        raise ValueError("A smoke run cannot define the retraining oracle")
    global_steps = int(source_metrics.get("global_steps", 0))
    if global_steps < 1:
        raise ValueError(
            "Source fine-tuning metrics must contain positive global_steps"
        )
    benchmark = source_metrics.get("benchmark", {})
    source_config = source_metrics.get("config", {})
    has_scheduler_steps = (
        isinstance(benchmark, Mapping)
        and "scheduler_total_steps" in benchmark
    )
    if (
        not has_scheduler_steps
        and isinstance(source_config, Mapping)
        and int(source_config.get("max_train_steps", -1)) > 0
    ):
        raise ValueError(
            "Source metrics predate scheduler recording and used a training "
            "step limit; the exact source schedule cannot be reconstructed"
        )
    scheduler_steps = (
        int(benchmark.get("scheduler_total_steps", global_steps))
        if isinstance(benchmark, Mapping)
        else global_steps
    )
    if scheduler_steps < global_steps:
        raise ValueError(
            "scheduler_total_steps cannot be smaller than global_steps"
        )
    return global_steps, scheduler_steps


def validate_retraining_split(split: Mapping[str, Any]) -> None:
    forget = set(int(value) for value in split.get("forget_indices", []))
    retain = set(int(value) for value in split.get("retain_train_indices", []))
    overlap = sorted(forget & retain)
    if overlap:
        raise ValueError(
            f"Retraining split leaks forgotten indices into retain_train: "
            f"{overlap[:10]}"
        )
    if not retain:
        raise ValueError("Retraining requires non-empty retain_train_indices")


def _validate_training_mode(cfg: FineTuneConfig) -> None:
    if cfg.training_mode not in {"finetune", "retrain_oracle"}:
        raise ValueError(
            "training_mode must be finetune or retrain_oracle"
        )
    if cfg.train_loader_key not in {"finetune_train", "retain_train"}:
        raise ValueError(
            "train_loader_key must be finetune_train or retain_train"
        )
    if cfg.training_mode == "retrain_oracle":
        if cfg.train_loader_key != "retain_train":
            raise ValueError(
                "Retraining oracle must use the retain_train loader"
            )
        if not cfg.initial_checkpoint:
            raise ValueError(
                "Retraining oracle requires the original base checkpoint"
            )
        if not cfg.source_metrics_path:
            raise ValueError(
                "Retraining oracle requires source fine-tuning metrics"
            )
        if not cfg.target_optimizer_steps or cfg.target_optimizer_steps < 1:
            raise ValueError(
                "Retraining oracle requires positive target_optimizer_steps"
            )
        if (
            cfg.target_scheduler_steps is not None
            and cfg.target_scheduler_steps < cfg.target_optimizer_steps
        ):
            raise ValueError(
                "target_scheduler_steps cannot be smaller than "
                "target_optimizer_steps"
            )


def validate_oracle_source_config(
    cfg: FineTuneConfig,
    source_metrics: Mapping[str, Any],
    initial_checkpoint_sha256: str,
) -> None:
    source_dataset = source_metrics.get("dataset")
    if source_dataset != cfg.dataset_name:
        raise ValueError(
            f"Oracle source dataset={source_dataset!r} does not match "
            f"{cfg.dataset_name!r}"
        )
    source_config = source_metrics.get("config")
    if not isinstance(source_config, Mapping):
        raise ValueError("Oracle source metrics are missing resolved config")
    fields = (
        "dataset_name",
        "model_name",
        "prompt_template",
        "adapter_rank",
        "adapter_alpha",
        "adapter_type",
        "lora_rank",
        "lora_alpha",
        "lora_layers",
        "lora_targets",
        "lora_dropout",
        "train_logit_scale",
        "precision",
        "gradient_checkpointing",
        "batch_size",
        "gradient_accumulation_steps",
        "lr",
        "weight_decay",
        "seed",
    )
    mismatches = {}
    for field in fields:
        if field not in source_config:
            continue
        source_value = source_config[field]
        current_value = getattr(cfg, field)
        if field == "lora_targets":
            source_value = tuple(source_value)
            current_value = tuple(current_value)
        if source_value != current_value:
            mismatches[field] = (source_value, current_value)
    if mismatches:
        raise ValueError(
            f"Oracle config differs from source fine-tuning run: {mismatches}"
        )
    expected_hash = source_metrics.get("base_checkpoint_sha256")
    if expected_hash and expected_hash != initial_checkpoint_sha256:
        raise ValueError(
            "Oracle initial checkpoint hash does not match source metrics"
        )


def resolve_training_plan(
    *,
    loader_batches: int,
    gradient_accumulation_steps: int,
    epochs: int,
    max_train_steps: int,
    target_optimizer_steps: int | None,
    target_scheduler_steps: int | None,
) -> tuple[int, int, int, int]:
    if loader_batches < 1:
        raise ValueError("Training loader must contain at least one batch")
    updates_per_epoch = math.ceil(
        loader_batches / gradient_accumulation_steps
    )
    natural_total_steps = epochs * updates_per_epoch
    target_steps = (
        target_optimizer_steps
        if target_optimizer_steps is not None
        else natural_total_steps
    )
    if max_train_steps > 0:
        target_steps = min(target_steps, max_train_steps)
    scheduler_steps = (
        target_scheduler_steps
        if target_scheduler_steps is not None
        else natural_total_steps
    )
    if scheduler_steps < target_steps:
        raise ValueError(
            "Scheduler steps cannot be smaller than optimizer steps"
        )
    epochs_to_run = (
        math.ceil(target_steps / updates_per_epoch)
        if target_optimizer_steps is not None
        else epochs
    )
    return updates_per_epoch, target_steps, scheduler_steps, epochs_to_run


def _validate_loaded_model_config(
    model: LightweightVLM,
    requested: ModelConfig,
) -> None:
    fields = (
        "model_name",
        "adapter_type",
        "adapter_rank",
        "adapter_alpha",
        "lora_rank",
        "lora_alpha",
        "lora_layers",
        "lora_targets",
        "lora_dropout",
        "train_logit_scale",
        "precision",
        "gradient_checkpointing",
    )
    mismatches = {}
    for field in fields:
        actual = getattr(model.cfg, field)
        expected = getattr(requested, field)
        if field == "lora_targets":
            actual = tuple(actual)
            expected = tuple(expected)
        if actual != expected:
            mismatches[field] = (actual, expected)
    if mismatches:
        raise ValueError(
            f"Initial checkpoint architecture does not match config: "
            f"{mismatches}"
        )


def _evaluate_all(
    model,
    loaders,
    class_text_inputs,
    device: torch.device,
    max_batches: int | None = None,
    non_blocking: bool = False,
    evaluate_test: bool = True,
) -> Dict[str, float]:
    model.eval()
    class_text_features = build_class_text_features(
        model, class_text_inputs, device
    )
    evaluation_args = {
        "class_text_inputs": class_text_inputs,
        "device": device,
        "class_text_features": class_text_features,
        "max_batches": max_batches,
        "non_blocking": non_blocking,
    }
    retain_val = evaluate_classification(
        model, loaders["retain_val"], **evaluation_args
    )
    metrics: Dict[str, float] = {
        "retain_val_acc": retain_val["accuracy"],
        "retain_val_loss": retain_val["loss"],
        "retain_val_macro_accuracy": retain_val["macro_accuracy"],
    }
    if not evaluate_test:
        return metrics
    test_all = evaluate_classification(
        model, loaders["test_all"], **evaluation_args
    )
    test_retain = evaluate_classification(
        model, loaders["test_retain"], **evaluation_args
    )
    forget_train = evaluate_classification(
        model, loaders["forget"], **evaluation_args
    )
    metrics.update(
        {
            "test_all_acc": test_all["accuracy"],
            "test_retain_acc": test_retain["accuracy"],
            "forget_train_acc": forget_train["accuracy"],
        }
    )
    return metrics


def run_finetuning(cfg: FineTuneConfig) -> Dict[str, str | float]:
    if cfg.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    _validate_training_mode(cfg)
    run_started = time.perf_counter()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    provenance = collect_run_provenance(device)
    resolved_config = asdict(cfg)

    output_dir = Path(cfg.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    split, dataset_spec, class_names = load_split_metadata(
        cfg.split_path, cfg.dataset_name
    )
    initial_checkpoint_sha256 = None
    source_metrics: Dict[str, Any] | None = None
    if cfg.training_mode == "retrain_oracle":
        validate_retraining_split(split)
        if cfg.source_metrics_path is None or cfg.initial_checkpoint is None:
            raise RuntimeError("Oracle source artifacts are missing")
        source_metrics = load_json(cfg.source_metrics_path)
        resolve_oracle_schedule(source_metrics)
        initial_checkpoint_sha256 = _file_sha256(cfg.initial_checkpoint)
        validate_oracle_source_config(
            cfg, source_metrics, initial_checkpoint_sha256
        )

    image_processor = CLIPImageProcessor.from_pretrained(
        cfg.model_name, local_files_only=cfg.local_files_only
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.model_name, local_files_only=cfg.local_files_only
    )
    class_text_inputs = build_canonical_prompt_inputs(
        tokenizer,
        class_names=class_names,
        dataset_name=dataset_spec.name,
    )
    prompt_contract_metadata = (
        {
            "version": class_text_inputs.contract.version,
            "digest": class_text_inputs.contract.digest,
            "template_count": class_text_inputs.template_count,
        }
        if hasattr(class_text_inputs, "contract")
        else {
            "version": "legacy_single_template",
            "digest": None,
            "template_count": 1,
        }
    )
    loaders = build_loaders(
        data_dir=cfg.data_dir,
        split_path=cfg.split_path,
        image_processor=image_processor,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        dataset_name=dataset_spec.name,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
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
        lora_dropout=cfg.lora_dropout,
        train_logit_scale=cfg.train_logit_scale,
        precision=cfg.precision,
        gradient_checkpointing=cfg.gradient_checkpointing,
        local_files_only=cfg.local_files_only,
    )

    initial_checkpoint_metadata: Dict[str, Any] | None = None
    if cfg.initial_checkpoint:
        model, initial_checkpoint_metadata = load_checkpoint(
            cfg.initial_checkpoint, map_location=device
        )
        validate_checkpoint_dataset(
            initial_checkpoint_metadata,
            dataset_spec.name,
            cfg.initial_checkpoint,
        )
        _validate_loaded_model_config(model, model_cfg)
        if initial_checkpoint_sha256 is None:
            initial_checkpoint_sha256 = _file_sha256(
                cfg.initial_checkpoint
            )
        model = model.to(device)
    else:
        model = LightweightVLM.from_config(model_cfg).to(device)
    setup_seconds = time.perf_counter() - run_started

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

    initial_checkpoint_name = (
        "oracle_init.pt"
        if cfg.training_mode == "retrain_oracle"
        else "base_init.pt"
    )
    initial_output_path = ckpt_dir / initial_checkpoint_name
    save_checkpoint(
        str(initial_output_path),
        model,
        extra={
            "stage": (
                "retrain_oracle_init"
                if cfg.training_mode == "retrain_oracle"
                else "base_init"
            ),
            "note": (
                "restored original adapter initialization"
                if cfg.initial_checkpoint
                else "randomly initialized adapters"
            ),
            "dataset": dataset_spec.name,
            "class_names": class_names,
            "architecture": model.architecture_summary(),
            "provenance": provenance,
            "training_config": resolved_config,
            "prompt_contract": prompt_contract_metadata,
            "smoke_mode": cfg.smoke_mode,
            "source_initial_checkpoint": cfg.initial_checkpoint,
            "source_initial_checkpoint_sha256": initial_checkpoint_sha256,
        },
    )
    initial_output_sha256 = _file_sha256(initial_output_path)

    optimizer = AdamW(model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_loader = loaders[cfg.train_loader_key]
    (
        updates_per_epoch,
        target_steps,
        scheduler_total_steps,
        epochs_to_run,
    ) = resolve_training_plan(
        loader_batches=len(train_loader),
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        epochs=cfg.epochs,
        max_train_steps=cfg.max_train_steps,
        target_optimizer_steps=cfg.target_optimizer_steps,
        target_scheduler_steps=cfg.target_scheduler_steps,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, scheduler_total_steps)
    )
    recovery_payload = None
    start_epoch = 0
    resumed_global_step = 0
    if cfg.resume_checkpoint:
        recovery_payload = load_recovery_checkpoint(
            cfg.resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
        )
        start_epoch = int(recovery_payload["epoch"])
        resumed_global_step = int(recovery_payload["global_step"])
    last_path = ckpt_dir / (
        "retrained_last.pt"
        if cfg.training_mode == "retrain_oracle"
        else "finetuned_last.pt"
    )
    recovery_path = ckpt_dir / "recovery_latest.pt"

    best_metric = -1.0
    best_path = ckpt_dir / (
        "retrained_best.pt"
        if cfg.training_mode == "retrain_oracle"
        else "finetuned_best.pt"
    )
    global_step = resumed_global_step
    processed_examples = 0
    gradient_parameter_count = 0
    training_started = time.perf_counter()
    optimization_seconds = 0.0
    evaluation_seconds = 0.0
    show_progress = os.environ.get("UNML_TQDM", "0") == "1"
    final_metrics: Dict[str, float] | None = None

    for epoch in range(start_epoch, epochs_to_run):
        model.set_train_mode()
        epoch_losses = []
        optimization_started = time.perf_counter()
        progress = tqdm(
            train_loader,
            desc=(
                f"{cfg.training_mode} epoch "
                f"{epoch+1}/{epochs_to_run}"
            ),
            disable=not show_progress,
        )
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(progress):
            batch = move_to_device(
                batch, device, non_blocking=cfg.non_blocking
            )
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
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite fine-tuning loss at epoch {epoch + 1}, "
                    f"optimizer step {global_step}"
                )

            (loss / cfg.gradient_accumulation_steps).backward()
            if gradient_parameter_count == 0:
                gradient_parameter_count = sum(
                    parameter.grad is not None
                    and bool(torch.count_nonzero(parameter.grad).item())
                    for parameter in model.trainable_parameters()
                )
            should_step = (
                (batch_index + 1) % cfg.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
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
                    f"[{cfg.training_mode}] epoch={epoch + 1} "
                    f"step={global_step} "
                    f"loss={tensor_to_float(loss):.4f}",
                    flush=True,
                )

            if (
                should_step
                and global_step >= target_steps
            ):
                break
        optimization_seconds += time.perf_counter() - optimization_started

        evaluation_started = time.perf_counter()
        eval_metrics = _evaluate_all(
            model,
            loaders,
            class_text_inputs,
            device,
            max_batches=cfg.max_eval_batches,
            non_blocking=cfg.non_blocking,
            evaluate_test=cfg.evaluate_test,
        )
        evaluation_seconds += time.perf_counter() - evaluation_started
        final_metrics = eval_metrics
        epoch_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        full_metrics = {"epoch": float(epoch + 1), "train_loss": epoch_loss, **eval_metrics}
        print(
            f"[{cfg.training_mode}] {format_metrics(full_metrics)}",
            flush=True,
        )

        log_finetune_epoch(cfg.__dict__, epoch + 1, epoch_loss, eval_metrics)

        if eval_metrics["retain_val_acc"] > best_metric:
            best_metric = eval_metrics["retain_val_acc"]
            save_checkpoint(
                str(best_path),
                model,
                extra={
                    "stage": (
                        "retrained_oracle"
                        if cfg.training_mode == "retrain_oracle"
                        else "finetuned"
                    ),
                    "checkpoint_role": "best",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "dataset": dataset_spec.name,
                    "class_names": class_names,
                    "architecture": model.architecture_summary(),
                    "provenance": provenance,
                    "training_config": resolved_config,
                    "prompt_contract": prompt_contract_metadata,
                    "smoke_mode": cfg.smoke_mode,
                    "source_initial_checkpoint": cfg.initial_checkpoint,
                    "source_initial_checkpoint_sha256": (
                        initial_checkpoint_sha256
                    ),
                    "source_metrics_path": cfg.source_metrics_path,
                    "metrics": eval_metrics,
                },
            )

        save_checkpoint(
            str(last_path),
            model,
            extra={
                "stage": (
                    "retrained_oracle_last"
                    if cfg.training_mode == "retrain_oracle"
                    else "finetuned_last"
                ),
                "checkpoint_role": "last",
                "epoch": epoch + 1,
                "global_step": global_step,
                "dataset": dataset_spec.name,
                "class_names": class_names,
                "architecture": model.architecture_summary(),
                "provenance": provenance,
                "training_config": resolved_config,
                "prompt_contract": prompt_contract_metadata,
                "smoke_mode": cfg.smoke_mode,
                "metrics": eval_metrics,
            },
        )
        save_recovery_checkpoint(
            str(recovery_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            epoch=epoch + 1,
            global_step=global_step,
            metrics=full_metrics,
            provenance={
                "dataset": dataset_spec.name,
                "split_path": cfg.split_path,
                "prompt_contract": prompt_contract_metadata,
                "training_config": resolved_config,
            },
        )
        if global_step >= target_steps:
            break

    training_seconds = time.perf_counter() - training_started
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    benchmark = {
        "setup_seconds": setup_seconds,
        "training_seconds": training_seconds,
        "optimization_seconds": optimization_seconds,
        "evaluation_seconds": evaluation_seconds,
        "processed_examples": processed_examples,
        "examples_per_second": (
            processed_examples / training_seconds if training_seconds > 0 else 0.0
        ),
        "optimization_examples_per_second": (
            processed_examples / optimization_seconds
            if optimization_seconds > 0
            else 0.0
        ),
        "peak_gpu_memory_mb": peak_memory_mb,
        "physical_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "effective_batch_size": (
            cfg.batch_size * cfg.gradient_accumulation_steps
        ),
        "device": str(device),
        "precision": cfg.precision,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.persistent_workers and cfg.num_workers > 0,
        "prefetch_factor": (
            cfg.prefetch_factor if cfg.num_workers > 0 else None
        ),
        "non_blocking": cfg.non_blocking,
        "train_loader_key": cfg.train_loader_key,
        "train_examples": len(train_loader.dataset),
        "updates_per_epoch": updates_per_epoch,
        "target_optimizer_steps": target_steps,
        "scheduler_total_steps": scheduler_total_steps,
    }
    if final_metrics is None:
        raise RuntimeError("Fine-tuning completed without an evaluation snapshot")
    if cfg.smoke_mode and gradient_parameter_count == 0:
        raise RuntimeError("Smoke test found no non-zero trainable gradients")
    checkpoint_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    checkpoint_payload_verified = {
        "model_config",
        "adapter_state_dict",
        "architecture",
        "extra",
    }.issubset(checkpoint_payload)
    if not checkpoint_payload_verified:
        raise RuntimeError(f"Checkpoint payload verification failed: {best_path}")
    metrics_path = metrics_dir / (
        "retrain_metrics.json"
        if cfg.training_mode == "retrain_oracle"
        else "finetune_metrics.json"
    )
    benchmark["total_seconds"] = time.perf_counter() - run_started
    update_checkpoint_extra(
        str(best_path),
        {
            "benchmark": benchmark,
            "provenance": provenance,
            "training_config": resolved_config,
        },
    )
    benchmark["checkpoint_size_bytes"] = best_path.stat().st_size
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
            "smoke_mode": cfg.smoke_mode,
            "max_eval_batches": cfg.max_eval_batches,
            "evaluation_is_partial": cfg.max_eval_batches is not None,
            "gradient_parameter_count": gradient_parameter_count,
            "checkpoint_payload_verified": checkpoint_payload_verified,
            "provenance": provenance,
            "config": resolved_config,
            "training_mode": cfg.training_mode,
            "source_metrics_path": cfg.source_metrics_path,
            "source_initial_checkpoint": cfg.initial_checkpoint,
            "source_initial_checkpoint_sha256": initial_checkpoint_sha256,
            "base_checkpoint_sha256": initial_output_sha256,
            "source_initial_metadata": initial_checkpoint_metadata,
            "forget_train_overlap_count": 0,
        },
        metrics_path,
    )

    log_finetune_summary(cfg.__dict__, best_metric, epochs_to_run)

    return {
        "base_checkpoint": str(initial_output_path),
        "best_checkpoint": str(best_path),
        "metrics_path": str(metrics_path),
        "best_retain_val_acc": best_metric,
    }
