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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune lightweight CLIP adapters on CIFAR-10")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split-path", type=str, default="outputs/splits/cifar10_split.json")
    parser.add_argument("--output-dir", type=str, default="outputs/finetune")

    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--prompt-template", type=str, default="a photo of a {}")
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--adapter-alpha", type=float, default=8.0)
    parser.add_argument("--no-train-logit-scale", action="store_true") # when this is called from run_pipeline.py, it isnt mentioned there, which causes its flag to be set as False, thats why we 'not' it below when setting the config. 
    # logit_scale is CLIP's learned temperature parameter (which is a sclar that controls how sharp the similarity distribution is b/w img and embedding. It acts as a softmax over class logits. In CLIP, the value is ~ln(100)=4.6. We train the logit_scale for both finetuning and unlearning. This essentially means that we give the model one more degree of freedom to adjust classification confidence. Freezing it would keep the output closer to CLIP behaviour. Im not sure if 'not freezing it' is the right call for the unlearning, because if one method shifts the logit_scale more than another, comparision becomes messy. This will have to be verified by freezing it during unlearning and see if MIA or forget quality change. If they dont, then the system is fine, but if they do then it becomes a temperature drift parameter to account for in adapter unlearning. The delta-based MIA is computed by (confidence_after_unlearning - confidence_base). If the logit_scale shifts during unlearning, the delta change reflects both unlearning and temperature change, which makes it harder to interpret which is genuine forgetting v/s the model changing its confidence due to temperature drift (temp drift as a confound in MIA eval for adapter-based unlearning).

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-steps", type=int, default=-1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from unml.train import FineTuneConfig, run_finetuning

    cfg = FineTuneConfig(
        data_dir=args.data_dir,
        split_path=args.split_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        prompt_template=args.prompt_template,
        adapter_rank=args.adapter_rank,
        adapter_alpha=args.adapter_alpha,
        train_logit_scale=not args.no_train_logit_scale,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        max_train_steps=args.max_train_steps,
        seed=args.seed,
        device=args.device,
    )
    result = run_finetuning(cfg)
    print(result)


if __name__ == "__main__":
    main()
