#!/usr/bin/env python3
"""Create a target-specific retain/forget split for a canonical oracle run."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from _bootstrap import REPO_ROOT, configure_runtime
except ModuleNotFoundError:
    from scripts._bootstrap import REPO_ROOT, configure_runtime

configure_runtime()

from unml.data import download_and_prepare_splits  # noqa: E402
from unml.data import summarize_splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--forget-classes", required=True)
    parser.add_argument("--forget-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    forget_classes = [int(value.strip()) for value in args.forget_classes.split(",") if value.strip()]
    download_and_prepare_splits(
        data_dir=str(args.data_dir),
        split_path=str(args.output_path),
        forget_classes=forget_classes,
        forget_fraction=args.forget_fraction,
        retain_val_fraction=0.1,
        seed=args.seed,
        dataset_name="cifar100",
        request_name=args.request,
    )
    print(f"[oracle-split] path={args.output_path}")
    print(f"[oracle-split] summary={summarize_splits(str(args.output_path))}")


if __name__ == "__main__":
    main()
