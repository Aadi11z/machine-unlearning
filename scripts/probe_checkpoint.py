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

from unml.config import (
    load_runtime_config,
    resolve_data_dir,
    resolve_dataset_and_split_path,
    resolve_model_value,
    resolve_output_root,
    resolve_section_str_list,
    resolve_stage_output_dir,
    resolve_value,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect selected CIFAR samples across model checkpoints"
    )
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "config" / "parameters.yaml")
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--request", default=None)
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--source", choices=("train", "test"), default=None)
    parser.add_argument("--index", type=int, action="append", default=[])
    parser.add_argument("--class-id", type=int, action="append", default=[])
    parser.add_argument("--class-name", action="append", default=[])
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--reference-checkpoint", default=None)
    parser.add_argument(
        "--candidate",
        action="append",
        help="Candidate checkpoint in the form name=path",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--save-images",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_cfg = load_runtime_config(args.config)
    dataset, split_path = resolve_dataset_and_split_path(
        args.dataset, args.split_path, runtime_cfg, args.request
    )
    training_dir = resolve_stage_output_dir(
        None, runtime_cfg, dataset, "training", args.request
    )
    unlearning_dir = resolve_stage_output_dir(
        None, runtime_cfg, dataset, "unlearning", args.request
    )
    reference_checkpoint = resolve_value(
        args.reference_checkpoint,
        runtime_cfg,
        ("probe", "reference_checkpoint"),
        str(training_dir / "checkpoints" / "finetuned_best.pt"),
    )

    candidate_names = []
    candidate_checkpoints = []
    if args.candidate:
        for entry in args.candidate:
            if "=" not in entry:
                raise ValueError(f"Invalid --candidate {entry!r}; use name=path")
            name, checkpoint = entry.split("=", 1)
            candidate_names.append(name)
            candidate_checkpoints.append(checkpoint)
    else:
        for method in resolve_section_str_list(
            None,
            runtime_cfg,
            dataset,
            "unlearning",
            "methods",
            ("counterfactual_rebind",),
        ):
            candidate_names.append(method)
            candidate_checkpoints.append(
                str(
                    unlearning_dir
                    / "checkpoints"
                    / f"unlearn_{method}.pt"
                )
            )

    from unml.probe import run_checkpoint_probe

    result = run_checkpoint_probe(
        data_dir=str(resolve_data_dir(args.data_dir, runtime_cfg)),
        split_path=split_path,
        dataset_name=dataset,
        source=resolve_value(
            args.source, runtime_cfg, ("probe", "source"), "test"
        ),
        indices=args.index,
        class_ids=args.class_id,
        class_names_requested=args.class_name,
        limit_per_class=resolve_value(
            args.limit_per_class,
            runtime_cfg,
            ("probe", "limit_per_class"),
            5,
        ),
        reference_checkpoint=reference_checkpoint,
        candidate_names=candidate_names,
        candidate_checkpoints=candidate_checkpoints,
        output_dir=str(
            Path(args.output_dir)
            if args.output_dir
            else resolve_output_root(
                None, runtime_cfg, dataset, args.request
            )
            / "probes"
        ),
        prompt_template=resolve_model_value(
            None,
            runtime_cfg,
            dataset,
            "prompt_template",
            "a photo of a {}",
        ),
        top_k=resolve_value(
            args.top_k, runtime_cfg, ("probe", "top_k"), 5
        ),
        device_name=resolve_value(
            args.device, runtime_cfg, ("experiment", "device"), "auto"
        ),
        save_images=resolve_value(
            args.save_images,
            runtime_cfg,
            ("probe", "save_images"),
            True,
        ),
        local_files_only=args.offline,
    )
    print(result)


if __name__ == "__main__":
    main()
