#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

from unml.config import (
    load_runtime_config,
    resolve_data_dir,
    resolve_dataset_and_split_path,
    resolve_model_value,
    resolve_request_name,
    resolve_stage_output_dir,
)
from unml.data import get_dataset_spec, load_split_metadata
from unml.preflight import (
    validate_cache_manifest,
    validate_cuda,
    validate_scratch_layout,
    validate_virtual_environment,
    validate_writable_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-fast validation before allocating time to a UNML run"
    )
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "config" / "parameters.yaml")
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--request", default=None)
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--require-venv",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--require-offline-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--scratch-project",
        default=os.environ.get("SCRATCH_PROJECT"),
        help="When set, require data, split, outputs, and HF cache under it",
    )
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_cfg = load_runtime_config(args.config)
    dataset_name, split_path = resolve_dataset_and_split_path(
        args.dataset, None, runtime_cfg, args.request
    )
    request_name = resolve_request_name(
        args.request, runtime_cfg, dataset_name
    )
    data_dir = resolve_data_dir(None, runtime_cfg).expanduser().resolve()
    split_path = Path(split_path).expanduser().resolve()
    output_dir = resolve_stage_output_dir(
        None, runtime_cfg, dataset_name, "training", request_name
    ).expanduser()
    model_name = str(
        resolve_model_value(
            None,
            runtime_cfg,
            dataset_name,
            "model_name",
            "openai/clip-vit-base-patch32",
        )
    )
    precision = str(
        resolve_model_value(
            None, runtime_cfg, dataset_name, "precision", "fp32"
        )
    )
    hf_home = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ).expanduser().resolve()

    split, dataset_spec, _ = load_split_metadata(
        str(split_path), dataset_name
    )
    stored_request = split.get("request_name")
    if request_name is not None and stored_request != request_name:
        raise ValueError(
            f"Split request={stored_request!r} does not match "
            f"requested request={request_name!r}"
        )

    # Instantiating without download verifies the extracted CIFAR files and
    # checks their archive integrity before GPU work starts.
    spec = get_dataset_spec(dataset_name)
    train_dataset = spec.dataset_class(
        root=str(data_dir), train=True, download=False
    )
    test_dataset = spec.dataset_class(
        root=str(data_dir), train=False, download=False
    )
    writable_output = validate_writable_directory(output_dir)
    scratch_root = validate_scratch_layout(
        {
            "data_dir": data_dir,
            "split_path": split_path,
            "output_dir": writable_output,
            "hf_home": hf_home,
        },
        args.scratch_project,
    )
    cache = None
    if args.require_offline_cache:
        manifest_path, manifest = validate_cache_manifest(
            model_name=model_name,
            dataset_name=dataset_name,
            hf_home=hf_home,
        )
        cache = {
            "manifest_path": str(manifest_path),
            "verified_at_utc": manifest["verified_at_utc"],
        }

    payload = {
        "status": "passed",
        "dataset": dataset_spec.name,
        "request": request_name,
        "model_name": model_name,
        "precision": precision,
        "data_dir": str(data_dir),
        "split_path": str(split_path),
        "output_dir": str(writable_output),
        "hf_home": str(hf_home),
        "scratch_project": str(scratch_root) if scratch_root else None,
        "train_examples": len(train_dataset),
        "test_examples": len(test_dataset),
        "forget_examples": len(split["forget_indices"]),
        "virtual_environment": validate_virtual_environment(
            args.require_venv
        ),
        "cuda": validate_cuda(args.require_cuda, precision),
        "cache": cache,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered, flush=True)
    if args.json_output:
        destination = Path(args.json_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
