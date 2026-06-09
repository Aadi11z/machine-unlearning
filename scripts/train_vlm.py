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
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_cfg = load_runtime_config(args.config)
    dataset_name, split_path = resolve_dataset_and_split_path(
        args.dataset, args.split_path, runtime_cfg, args.request
    )
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

    from unml.train import FineTuneConfig, run_finetuning

    cfg = FineTuneConfig(
        data_dir=str(resolve_data_dir(args.data_dir, runtime_cfg)),
        split_path=split_path,
        output_dir=str(
            resolve_stage_output_dir(
                args.output_dir,
                runtime_cfg,
                dataset_name,
                "training",
                args.request,
            )
        ),
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
        num_workers=resolve_value(
            args.num_workers, runtime_cfg, ("training", "num_workers"), 4
        ),
        lr=resolve_value(args.lr, runtime_cfg, ("training", "lr"), 3e-3),
        weight_decay=resolve_value(
            args.weight_decay, runtime_cfg, ("training", "weight_decay"), 1e-4
        ),
        epochs=resolve_value(args.epochs, runtime_cfg, ("training", "epochs"), 5),
        max_train_steps=resolve_value(
            args.max_train_steps, runtime_cfg, ("training", "max_train_steps"), -1
        ),
        seed=resolve_value(args.seed, runtime_cfg, ("experiment", "seed"), 42),
        device=resolve_value(
            args.device, runtime_cfg, ("experiment", "device"), "auto"
        ),
    )
    result = run_finetuning(cfg)
    print(result)


if __name__ == "__main__":
    main()
