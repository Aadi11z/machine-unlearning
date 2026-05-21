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

from unml.data import download_and_prepare_splits, summarize_splits
from unml.config import load_runtime_config, parse_forget_classes, resolve_value
from unml.utils import format_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CIFAR-10 and create unlearning splits"
    )
    parser.add_argument(
        "--config", type=str, default=str(REPO_ROOT / "config" / "parameters.yaml")
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--split-path", type=str, default=None)
    parser.add_argument(
        "--forget-classes",
        type=str,
        default=None,
        help="Comma-separated class ids to forget (e.g. 1,2)",
    )
    parser.add_argument("--forget-fraction", type=float, default=None)
    parser.add_argument("--retain-val-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_cfg = load_runtime_config(args.config)

    data_dir = resolve_value(args.data_dir, runtime_cfg, ("data", "data_dir"), "data")
    split_path = resolve_value(
        args.split_path,
        runtime_cfg,
        ("data", "split_path"),
        "outputs/splits/cifar10_split.json",
    )
    forget_classes_raw = resolve_value(
        args.forget_classes, runtime_cfg, ("data", "forget_classes"), "3,5"
    )
    forget_fraction = resolve_value(
        args.forget_fraction, runtime_cfg, ("data", "forget_fraction"), 1.0
    )
    retain_val_fraction = resolve_value(
        args.retain_val_fraction, runtime_cfg, ("data", "retain_val_fraction"), 0.1
    )
    seed = resolve_value(args.seed, runtime_cfg, ("experiment", "seed"), 42)

    forget_classes = parse_forget_classes(forget_classes_raw)

    download_and_prepare_splits(
        data_dir=data_dir,
        split_path=split_path,
        forget_classes=forget_classes,
        forget_fraction=forget_fraction,
        retain_val_fraction=retain_val_fraction,
        seed=seed,
    )

    summary = summarize_splits(split_path)
    print(f"[splits] {format_metrics(summary)}")
    print(f"[splits] forget_classes={forget_classes}")
    print(f"[splits] split_path={split_path}")


if __name__ == "__main__":
    main()
