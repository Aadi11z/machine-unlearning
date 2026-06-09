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

from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

from unml.config import (
    load_runtime_config,
    normalize_dataset_name,
    resolve_model_value,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify CLIP artifacts before requesting a GPU"
    )
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "config" / "parameters.yaml")
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_cfg = load_runtime_config(args.config)
    dataset_name = normalize_dataset_name(
        args.dataset or runtime_cfg.get("data", {}).get("dataset", "cifar10")
    )
    model_name = resolve_model_value(
        args.model_name,
        runtime_cfg,
        dataset_name,
        "model_name",
        "openai/clip-vit-base-patch32",
    )

    print(f"[cache] dataset={dataset_name}", flush=True)
    print(f"[cache] model={model_name}", flush=True)
    print(f"[cache] HF_HOME={os.environ.get('HF_HOME', '<default>')}", flush=True)

    CLIPImageProcessor.from_pretrained(model_name)
    CLIPTokenizer.from_pretrained(model_name)
    CLIPModel.from_pretrained(model_name)

    # Verify the complete artifact set is usable without network access.
    CLIPImageProcessor.from_pretrained(model_name, local_files_only=True)
    CLIPTokenizer.from_pretrained(model_name, local_files_only=True)
    CLIPModel.from_pretrained(model_name, local_files_only=True)
    print("[cache] offline verification passed", flush=True)


if __name__ == "__main__":
    main()
