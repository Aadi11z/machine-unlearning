#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from unml.config import (
    load_runtime_config,
    resolve_data_dir,
    resolve_dataset_and_split_path,
    resolve_model_value,
    resolve_section_value,
    resolve_stage_output_dir,
    resolve_value,
)
from unml.utils import transformers_offline
from unml.utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune lightweight CLIP adapters on CIFAR-10/CIFAR-100"
    )
    parser.add_argument(
        "--config", type=str, default=str(REPO_ROOT / "config" / "parameters.yaml")
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--request", type=str, default=None)
    parser.add_argument("--split-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--oracle",
        action="store_true",
        help=(
            "Retrain from the original adapter initialization using only "
            "retain_train and the source optimizer schedule"
        ),
    )
    parser.add_argument("--initial-checkpoint", type=str, default=None)
    parser.add_argument("--source-metrics", type=str, default=None)
    parser.add_argument("--target-optimizer-steps", type=int, default=None)
    parser.add_argument("--target-scheduler-steps", type=int, default=None)

    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--prompt-template", type=str, default=None)
    parser.add_argument("--adapter-rank", type=int, default=None)
    parser.add_argument("--adapter-alpha", type=float, default=None)
    parser.add_argument(
        "--adapter-type",
        type=str,
        default=None,
        choices=("post_projection", "vision_lora"),
    )
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--lora-layers", type=str, default=None)
    parser.add_argument("--lora-targets", type=str, default=None)
    parser.add_argument(
        "--precision", type=str, default=None, choices=("fp32", "fp16", "bf16")
    )
    checkpointing_group = parser.add_mutually_exclusive_group()
    checkpointing_group.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
    )
    checkpointing_group.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.set_defaults(gradient_checkpointing=None)
    logit_group = parser.add_mutually_exclusive_group()
    logit_group.add_argument(
        "--train-logit-scale", dest="train_logit_scale", action="store_true"
    )
    logit_group.add_argument(
        "--no-train-logit-scale", dest="train_logit_scale", action="store_false"
    )
    parser.set_defaults(train_logit_scale=None)
    # when this is called from run_pipeline.py, it isnt mentioned there, which causes its flag to be set as False, thats why we 'not' it below when setting the config.
    # logit_scale is CLIP's learned temperature parameter (which is a sclar that controls how sharp the similarity distribution is b/w img and embedding. It acts as a softmax over class logits. In CLIP, the value is ~ln(100)=4.6. We train the logit_scale for both finetuning and unlearning. This essentially means that we give the model one more degree of freedom to adjust classification confidence. Freezing it would keep the output closer to CLIP behaviour. Im not sure if 'not freezing it' is the right call for the unlearning, because if one method shifts the logit_scale more than another, comparision becomes messy. This will have to be verified by freezing it during unlearning and see if MIA or forget quality change. If they dont, then the system is fine, but if they do then it becomes a temperature drift parameter to account for in adapter unlearning. The delta-based MIA is computed by (confidence_after_unlearning - confidence_base). If the logit_scale shifts during unlearning, the delta change reflects both unlearning and temperature change, which makes it harder to interpret which is genuine forgetting v/s the model changing its confidence due to temperature drift (temp drift as a confound in MIA eval for adapter-based unlearning).

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument(
        "--non-blocking",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the configured short benchmark with bounded evaluation",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require model/tokenizer/processor artifacts to exist in cache",
    )

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_cfg = load_runtime_config(args.config)
    dataset_name, split_path = resolve_dataset_and_split_path(
        args.dataset, args.split_path, runtime_cfg, args.request
    )
    if args.oracle and args.smoke:
        raise ValueError("--oracle and --smoke cannot be combined")
    lora_targets = resolve_model_value(
        args.lora_targets,
        runtime_cfg,
        dataset_name,
        "lora_targets",
        ["q_proj", "v_proj"],
    )
    if isinstance(lora_targets, str):
        lora_targets = [
            target.strip() for target in lora_targets.split(",") if target.strip()
        ]
    training_output_dir = resolve_stage_output_dir(
        None,
        runtime_cfg,
        dataset_name,
        "training",
        args.request,
    )
    default_output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            training_output_dir.parent / "retrain_oracle"
            if args.oracle
            else training_output_dir
        )
    )
    if args.smoke and args.output_dir is None:
        default_output_dir = default_output_dir.parent / "phase2_benchmark"

    smoke_epochs = resolve_value(
        None, runtime_cfg, ("training", "smoke", "epochs"), 1
    )
    smoke_steps = resolve_value(
        None, runtime_cfg, ("training", "smoke", "max_train_steps"), 50
    )
    smoke_eval_batches = resolve_value(
        None, runtime_cfg, ("training", "smoke", "max_eval_batches"), 2
    )

    from unml.train import (
        FineTuneConfig,
        resolve_oracle_schedule,
        run_finetuning,
    )

    initial_checkpoint = None
    source_metrics_path = None
    target_optimizer_steps = None
    target_scheduler_steps = None
    if args.oracle:
        initial_checkpoint = resolve_value(
            args.initial_checkpoint,
            runtime_cfg,
            ("retraining", "initial_checkpoint"),
            str(training_output_dir / "checkpoints" / "base_init.pt"),
        )
        source_metrics_path = resolve_value(
            args.source_metrics,
            runtime_cfg,
            ("retraining", "source_metrics"),
            str(training_output_dir / "metrics" / "finetune_metrics.json"),
        )
        source_metrics = load_json(source_metrics_path)
        source_optimizer_steps, source_scheduler_steps = (
            resolve_oracle_schedule(source_metrics)
        )
        target_optimizer_steps = resolve_value(
            args.target_optimizer_steps,
            runtime_cfg,
            ("retraining", "target_optimizer_steps"),
            source_optimizer_steps,
        )
        target_scheduler_steps = resolve_value(
            args.target_scheduler_steps,
            runtime_cfg,
            ("retraining", "target_scheduler_steps"),
            source_scheduler_steps,
        )

    cfg = FineTuneConfig(
        data_dir=str(resolve_data_dir(args.data_dir, runtime_cfg)),
        split_path=split_path,
        output_dir=str(default_output_dir),
        dataset_name=dataset_name,
        model_name=resolve_model_value(
            args.model_name,
            runtime_cfg,
            dataset_name,
            "model_name",
            "openai/clip-vit-base-patch32",
        ),
        prompt_template=resolve_model_value(
            args.prompt_template,
            runtime_cfg,
            dataset_name,
            "prompt_template",
            "a photo of a {}",
        ),
        adapter_rank=resolve_model_value(
            args.adapter_rank, runtime_cfg, dataset_name, "adapter_rank", 8
        ),
        adapter_alpha=resolve_model_value(
            args.adapter_alpha, runtime_cfg, dataset_name, "adapter_alpha", 8.0
        ),
        adapter_type=resolve_model_value(
            args.adapter_type,
            runtime_cfg,
            dataset_name,
            "adapter_type",
            "post_projection",
        ),
        lora_rank=resolve_model_value(
            args.lora_rank, runtime_cfg, dataset_name, "lora_rank", 8
        ),
        lora_alpha=resolve_model_value(
            args.lora_alpha, runtime_cfg, dataset_name, "lora_alpha", 8.0
        ),
        lora_layers=resolve_model_value(
            args.lora_layers, runtime_cfg, dataset_name, "lora_layers", "all"
        ),
        lora_targets=lora_targets,
        train_logit_scale=resolve_model_value(
            args.train_logit_scale,
            runtime_cfg,
            dataset_name,
            "train_logit_scale",
            True,
        ),
        precision=resolve_model_value(
            args.precision, runtime_cfg, dataset_name, "precision", "fp32"
        ),
        gradient_checkpointing=resolve_model_value(
            args.gradient_checkpointing,
            runtime_cfg,
            dataset_name,
            "gradient_checkpointing",
            False,
        ),
        batch_size=resolve_section_value(
            args.batch_size,
            runtime_cfg,
            dataset_name,
            "training",
            "batch_size",
            128,
        ),
        gradient_accumulation_steps=resolve_section_value(
            args.gradient_accumulation_steps,
            runtime_cfg,
            dataset_name,
            "training",
            "gradient_accumulation_steps",
            1,
        ),
        num_workers=resolve_section_value(
            args.num_workers,
            runtime_cfg,
            dataset_name,
            "training",
            "num_workers",
            4,
        ),
        pin_memory=resolve_section_value(
            args.pin_memory,
            runtime_cfg,
            dataset_name,
            "training",
            "pin_memory",
            True,
        ),
        persistent_workers=resolve_section_value(
            args.persistent_workers,
            runtime_cfg,
            dataset_name,
            "training",
            "persistent_workers",
            False,
        ),
        prefetch_factor=resolve_section_value(
            args.prefetch_factor,
            runtime_cfg,
            dataset_name,
            "training",
            "prefetch_factor",
            2,
        ),
        non_blocking=resolve_section_value(
            args.non_blocking,
            runtime_cfg,
            dataset_name,
            "training",
            "non_blocking",
            False,
        ),
        lr=resolve_value(args.lr, runtime_cfg, ("training", "lr"), 3e-3),
        weight_decay=resolve_value(
            args.weight_decay, runtime_cfg, ("training", "weight_decay"), 1e-4
        ),
        epochs=(
            resolve_value(args.epochs, runtime_cfg, ("training", "epochs"), 5)
            if not args.smoke or args.epochs is not None
            else smoke_epochs
        ),
        max_train_steps=(
            resolve_value(
                args.max_train_steps,
                runtime_cfg,
                ("training", "max_train_steps"),
                -1,
            )
            if not args.smoke or args.max_train_steps is not None
            else smoke_steps
        ),
        seed=resolve_value(args.seed, runtime_cfg, ("experiment", "seed"), 42),
        device=resolve_value(
            args.device, runtime_cfg, ("experiment", "device"), "auto"
        ),
        local_files_only=args.offline or transformers_offline(),
        max_eval_batches=(
            args.max_eval_batches
            if args.max_eval_batches is not None
            else smoke_eval_batches if args.smoke else None
        ),
        smoke_mode=args.smoke,
        training_mode="retrain_oracle" if args.oracle else "finetune",
        train_loader_key="retain_train" if args.oracle else "finetune_train",
        initial_checkpoint=initial_checkpoint,
        source_metrics_path=source_metrics_path,
        target_optimizer_steps=target_optimizer_steps,
        target_scheduler_steps=target_scheduler_steps,
    )
    result = run_finetuning(cfg)
    print(result)


if __name__ == "__main__":
    main()
