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
    parser = argparse.ArgumentParser(description="Run attack suite and compare utility/forget quality")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split-path", type=str, default="outputs/splits/cifar10_split.json")
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--base-checkpoint", type=str, required=True)

    parser.add_argument(
        "--candidate",
        type=str,
        action="append",
        required=True,
        help="Candidate checkpoint in the form name=path",
    )

    parser.add_argument("--output-dir", type=str, default="outputs/comparison")
    parser.add_argument("--prompt-template", type=str, default="a photo of a {}")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-attack-samples", type=int, default=4000)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from unml.attacks import AttackConfig, run_attack_comparison

    names = []
    checkpoints = []
    for entry in args.candidate:
        if "=" not in entry:
            raise ValueError(f"Invalid --candidate '{entry}'. Use name=path")
        name, path = entry.split("=", 1)
        names.append(name)
        checkpoints.append(path)

    cfg = AttackConfig(
        data_dir=args.data_dir,
        split_path=args.split_path,
        model_name=args.model_name,
        base_checkpoint=args.base_checkpoint,
        candidate_checkpoints=checkpoints,
        candidate_names=names,
        output_dir=args.output_dir,
        prompt_template=args.prompt_template,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_attack_samples=args.max_attack_samples,
        device=args.device,
    )

    result = run_attack_comparison(cfg)
    print(result)


if __name__ == "__main__":
    main()
