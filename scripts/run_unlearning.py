#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

UNLEARNING_METHODS = ("retain_only", "ga_kl", "counterfactual_rebind", "entropy_rebind")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run machine unlearning on fine-tuned lightweight VLM")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split-path", type=str, default="outputs/splits/cifar10_split.json")
    parser.add_argument("--finetuned-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/unlearning")

    parser.add_argument(
        "--method",
        type=str,
        default="counterfactual_rebind",
        choices=sorted(UNLEARNING_METHODS),
        help="Unlearning method",
    )
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--prompt-template", type=str, default="a photo of a {}")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--kl-temperature", type=float, default=1.5)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--ga-weight", type=float, default=1.0)
    parser.add_argument("--cf-weight", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--entropy-weight", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from unml.unlearn import UnlearnConfig, run_unlearning

    cfg = UnlearnConfig(
        data_dir=args.data_dir,
        split_path=args.split_path,
        finetuned_checkpoint=args.finetuned_checkpoint,
        output_dir=args.output_dir,
        method=args.method,
        model_name=args.model_name,
        prompt_template=args.prompt_template,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        kl_temperature=args.kl_temperature,
        kl_weight=args.kl_weight,
        ga_weight=args.ga_weight,
        cf_weight=args.cf_weight,
        margin_weight=args.margin_weight,
        margin=args.margin,
        entropy_weight=args.entropy_weight,
    )
    result = run_unlearning(cfg)
    print(result)


if __name__ == "__main__":
    main()
