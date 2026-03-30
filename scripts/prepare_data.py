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

from unml.data import download_and_prepare_splits, summarize_splits
from unml.utils import format_metrics

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CIFAR-10 and create unlearning splits")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split-path", type=str, default="outputs/splits/cifar10_split.json")
    parser.add_argument("--forget-classes", type=str, default="3,5", help="Comma-separated class ids to forget (e.g. 1,2)")
    parser.add_argument("--forget-fraction", type=float, default=1.0)
    parser.add_argument("--retain-val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()
# this is a test

def main() -> None:
    args = parse_args()

    forget_classes = [int(x.strip()) for x in args.forget_classes.split(",") if x.strip()] # "3, 5" to [3, 5]

    download_and_prepare_splits(
        data_dir=args.data_dir,
        split_path=args.split_path,
        forget_classes=forget_classes,
        forget_fraction=args.forget_fraction,
        retain_val_fraction=args.retain_val_fraction,
        seed=args.seed,
    )

    summary = summarize_splits(args.split_path)
    print(f"[splits] {format_metrics(summary)}")
    print(f"[splits] forget_classes={forget_classes}")
    print(f"[splits] split_path={args.split_path}")

if __name__ == "__main__":
    main()
