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

from unml.config import load_runtime_config, resolve_str_list, resolve_value

UNLEARNING_METHODS = ("retain_only", "ga_kl", "counterfactual_rebind", "entropy_rebind")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run machine unlearning on fine-tuned lightweight VLM"
    )
    parser.add_argument(
        "--config", type=str, default=str(REPO_ROOT / "config" / "parameters.yaml")
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--split-path", type=str, default=None)
    parser.add_argument("--finetuned-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=sorted(UNLEARNING_METHODS),
        help="Unlearning method",
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--prompt-template", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)

    parser.add_argument("--kl-temperature", type=float, default=None)
    parser.add_argument("--kl-weight", type=float, default=None)
    parser.add_argument("--ga-weight", type=float, default=None)
    parser.add_argument("--cf-weight", type=float, default=None)
    parser.add_argument("--margin-weight", type=float, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--entropy-weight", type=float, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_cfg = load_runtime_config(args.config)

    from unml.unlearn import UnlearnConfig, run_unlearning

    methods = resolve_str_list(
        None, runtime_cfg, ("unlearning", "methods"), ("counterfactual_rebind",)
    )
    default_method = methods[0] if methods else "counterfactual_rebind"
    training_output_dir = resolve_value(
        None, runtime_cfg, ("training", "output_dir"), "outputs/finetune"
    )
    finetuned_checkpoint = resolve_value(
        args.finetuned_checkpoint,
        runtime_cfg,
        ("unlearning", "finetuned_checkpoint"),
        str(Path(training_output_dir) / "checkpoints" / "finetuned_best.pt"),
    )

    cfg = UnlearnConfig(
        data_dir=resolve_value(
            args.data_dir, runtime_cfg, ("data", "data_dir"), "data"
        ),
        split_path=resolve_value(
            args.split_path,
            runtime_cfg,
            ("data", "split_path"),
            "outputs/splits/cifar10_split.json",
        ),
        finetuned_checkpoint=finetuned_checkpoint,
        output_dir=resolve_value(
            args.output_dir,
            runtime_cfg,
            ("unlearning", "output_dir"),
            "outputs/unlearning",
        ),
        method=resolve_value(
            args.method, runtime_cfg, ("unlearning", "method"), default_method
        ),
        model_name=resolve_value(
            args.model_name,
            runtime_cfg,
            ("model", "model_name"),
            "openai/clip-vit-base-patch32",
        ),
        prompt_template=resolve_value(
            args.prompt_template,
            runtime_cfg,
            ("model", "prompt_template"),
            "a photo of a {}",
        ),
        batch_size=resolve_value(
            args.batch_size, runtime_cfg, ("unlearning", "batch_size"), 128
        ),
        num_workers=resolve_value(
            args.num_workers, runtime_cfg, ("unlearning", "num_workers"), 4
        ),
        steps=resolve_value(args.steps, runtime_cfg, ("unlearning", "steps"), 500),
        lr=resolve_value(args.lr, runtime_cfg, ("unlearning", "lr"), 5e-4),
        weight_decay=resolve_value(
            args.weight_decay, runtime_cfg, ("unlearning", "weight_decay"), 0.0
        ),
        seed=resolve_value(args.seed, runtime_cfg, ("experiment", "seed"), 42),
        device=resolve_value(
            args.device, runtime_cfg, ("experiment", "device"), "auto"
        ),
        kl_temperature=resolve_value(
            args.kl_temperature, runtime_cfg, ("unlearning", "kl_temperature"), 1.5
        ),
        kl_weight=resolve_value(
            args.kl_weight, runtime_cfg, ("unlearning", "kl_weight"), 1.0
        ),
        ga_weight=resolve_value(
            args.ga_weight, runtime_cfg, ("unlearning", "ga_weight"), 1.0
        ),
        cf_weight=resolve_value(
            args.cf_weight, runtime_cfg, ("unlearning", "cf_weight"), 1.0
        ),
        margin_weight=resolve_value(
            args.margin_weight, runtime_cfg, ("unlearning", "margin_weight"), 0.5
        ),
        margin=resolve_value(args.margin, runtime_cfg, ("unlearning", "margin"), 0.2),
        entropy_weight=resolve_value(
            args.entropy_weight, runtime_cfg, ("unlearning", "entropy_weight"), 1.0
        ),
    )
    result = run_unlearning(cfg)
    print(result)


if __name__ == "__main__":
    main()
