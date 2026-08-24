#!/usr/bin/env python3
"""Generate the fixed target-neutral CIFAR-100 development split."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from _bootstrap import REPO_ROOT, configure_runtime
except ModuleNotFoundError:
    from scripts._bootstrap import REPO_ROOT, configure_runtime

configure_runtime()

from unml.canonical_split import generate_canonical_cifar100_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the fixed seed-42, stratified 45k/5k CIFAR-100 "
            "development split. This command accepts no forget request."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / "cifar100"
            / "canonical"
            / "development"
            / "splits"
            / "cifar100_canonical_development_v1.json"
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR-100 if it is not already present in --data-dir",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing split artifact after regenerating and validating it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = generate_canonical_cifar100_split(
        data_dir=args.data_dir,
        output_path=args.output_path,
        download=args.download,
        overwrite=args.overwrite,
    )
    print(f"[canonical-split] path={args.output_path}")
    print(f"[canonical-split] split_id={payload['split_id']}")
    print(f"[canonical-split] digest={payload['digest']}")
    print("[canonical-split] development=45000 train / 5000 validation")
    print("[canonical-split] official_test=10000 untouched examples")


if __name__ == "__main__":
    main()
