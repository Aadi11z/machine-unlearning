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
from unml.config import (
    load_runtime_config,
    parse_forget_classes,
    parse_optional_classes,
    resolve_data_dir,
    resolve_dataset_and_split_path,
    resolve_dataset_value,
    resolve_forget_classes,
    resolve_request_config,
    resolve_request_name,
    resolve_value,
)
from unml.utils import format_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CIFAR-10/CIFAR-100 and create unlearning splits"
    )
    parser.add_argument(
        "--config", type=str, default=str(REPO_ROOT / "config" / "parameters.yaml")
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--request", type=str, default=None)
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
    dataset_name, split_path = resolve_dataset_and_split_path(
        args.dataset, args.split_path, runtime_cfg, args.request
    )
    request_name = resolve_request_name(args.request, runtime_cfg, dataset_name)
    request_cfg = resolve_request_config(args.request, runtime_cfg, dataset_name)

    data_dir = str(resolve_data_dir(args.data_dir, runtime_cfg))
    forget_classes_raw = resolve_forget_classes(
        args.forget_classes, runtime_cfg, dataset_name, args.request
    )
    forget_fraction = resolve_dataset_value(
        args.forget_fraction,
        runtime_cfg,
        dataset_name,
        "forget_fraction",
        1.0,
    )
    retain_val_fraction = resolve_dataset_value(
        args.retain_val_fraction,
        runtime_cfg,
        dataset_name,
        "retain_val_fraction",
        0.1,
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
        dataset_name=dataset_name,
        request_name=request_name,
        request_type=(
            str(request_cfg["request_type"]) if request_cfg is not None else None
        ),
        superclass=(
            str(request_cfg["superclass"]) if request_cfg is not None else None
        ),
        sibling_classes=parse_optional_classes(
            request_cfg.get("sibling_classes") if request_cfg is not None else None
        ),
    )

    summary = summarize_splits(split_path)
    print(f"[splits] {format_metrics(summary)}")
    print(f"[splits] dataset={dataset_name}")
    print(f"[splits] request={request_name or 'class_list'}")
    print(f"[splits] forget_classes={forget_classes}")
    print(f"[splits] split_path={split_path}")


if __name__ == "__main__":
    main()
