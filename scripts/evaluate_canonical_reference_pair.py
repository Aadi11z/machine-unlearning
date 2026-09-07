#!/usr/bin/env python3
"""Evaluate canonical and oracle checkpoints on one identical oracle test split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from _bootstrap import configure_runtime
except ModuleNotFoundError:
    from scripts._bootstrap import configure_runtime

configure_runtime()

import torch  # noqa: E402
from transformers import CLIPImageProcessor, CLIPTokenizer  # noqa: E402

from unml.data import (  # noqa: E402
    build_canonical_prompt_inputs,
    build_loaders,
    load_split_metadata,
    validate_checkpoint_dataset,
)
from unml.evaluate import build_class_text_features, evaluate_classification  # noqa: E402
from unml.manifest import sha256_file, verify_manifest_artifacts  # noqa: E402
from unml.model import load_checkpoint  # noqa: E402
from unml.utils import get_device, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    canonical_dir = args.canonical_dir.resolve()
    oracle_dir = args.oracle_dir.resolve()
    canonical_manifest_path = canonical_dir / "manifest.json"
    canonical = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
    verify_manifest_artifacts(canonical, root=canonical_dir)
    split_candidates = sorted((oracle_dir / "splits").glob("*.json"))
    if len(split_candidates) != 1:
        raise RuntimeError("Reference evaluation requires exactly one oracle split")
    split_path = split_candidates[0]
    split, dataset_spec, class_names = load_split_metadata(split_path, "cifar100")
    model_name = str(canonical["model_config"]["model_name"])
    image_processor = CLIPImageProcessor.from_pretrained(
        model_name, local_files_only=args.offline
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        model_name, local_files_only=args.offline
    )
    class_text_inputs = build_canonical_prompt_inputs(
        tokenizer, class_names=class_names, dataset_name=dataset_spec.name
    )
    loaders = build_loaders(
        data_dir=str(args.data_dir),
        split_path=str(split_path),
        image_processor=image_processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_name=dataset_spec.name,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2,
    )
    device = get_device(args.device)
    canonical_checkpoint = (
        canonical_dir / canonical["artifacts"]["checkpoint"]["path"]
    )
    oracle_checkpoint = oracle_dir / "checkpoints" / "retrained_best.pt"
    evaluations = {
        "canonical": _evaluate_checkpoint(
            canonical_checkpoint,
            loaders=loaders,
            class_text_inputs=class_text_inputs,
            dataset_name=dataset_spec.name,
            device=device,
        ),
        "oracle": _evaluate_checkpoint(
            oracle_checkpoint,
            loaders=loaders,
            class_text_inputs=class_text_inputs,
            dataset_name=dataset_spec.name,
            device=device,
        ),
    }
    payload = {
        "schema": "unml-reference-comparison-v1",
        "dataset": dataset_spec.name,
        "baseline_id": canonical["baseline_id"],
        "oracle_id": "cifar100_retraining_oracle_v1",
        "request_name": split["request_name"],
        "request_type": split["request_type"],
        "forget_classes": split["forget_classes"],
        "prompt_contract": canonical["prompt_contract"],
        "split_sha256": sha256_file(split_path),
        "checkpoints": {
            "canonical_sha256": sha256_file(canonical_checkpoint),
            "oracle_sha256": sha256_file(oracle_checkpoint),
        },
        "evaluations": evaluations,
    }
    destination = oracle_dir / "metrics" / "reference_comparison.json"
    save_json(payload, destination)
    print(f"[reference-eval] comparison={destination}")
    print(json.dumps(evaluations, sort_keys=True))


def _evaluate_checkpoint(
    checkpoint: Path,
    *,
    loaders,
    class_text_inputs,
    dataset_name: str,
    device: torch.device,
) -> dict:
    model, metadata = load_checkpoint(str(checkpoint), map_location=device)
    validate_checkpoint_dataset(metadata, dataset_name, str(checkpoint))
    model = model.to(device).eval()
    text_features = build_class_text_features(model, class_text_inputs, device)
    result = {}
    for name in ("test_all", "test_retain", "test_forget"):
        result[name] = evaluate_classification(
            model,
            loaders[name],
            class_text_inputs,
            device,
            class_text_features=text_features,
            non_blocking=device.type == "cuda",
        )
    del model, text_features
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    main()
