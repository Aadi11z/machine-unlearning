#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact retained-data adapter retraining from the original "
            "fine-tuning initialization"
        )
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "parameters.yaml"),
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--request", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_vlm.py"),
        "--config",
        args.config,
        "--oracle",
    ]
    if args.dataset:
        command.extend(["--dataset", args.dataset])
    if args.request:
        command.extend(["--request", args.request])
    if args.device:
        command.extend(["--device", args.device])
    if args.offline:
        command.append("--offline")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
