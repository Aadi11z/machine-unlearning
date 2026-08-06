#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from _bootstrap import REPO_ROOT, configure_runtime
except ModuleNotFoundError:
    from scripts._bootstrap import REPO_ROOT, configure_runtime

configure_runtime()

from unml.probe_ui import create_probe_app, load_probe_ui_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local CIFAR-100 checkpoint probe interface"
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "parameters.yaml"),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Trusted startup override for the registered output root",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require locally cached CLIP assets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    settings = load_probe_ui_settings(
        args.config,
        output_root_override=args.output_root,
    )
    app = create_probe_app(
        settings,
        device_name=args.device,
        local_files_only=args.offline,
    )
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
