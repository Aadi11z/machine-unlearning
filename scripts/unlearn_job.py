#!/usr/bin/env python3
"""Headless single-class unlearning job runner.

Builds the split for any CIFAR-100 class on demand and runs one unlearning
method against a fixed fine-tuned baseline, producing a job_result.json that
downstream platforms can consume without reading repository configuration.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from _bootstrap import REPO_ROOT, configure_runtime
except ModuleNotFoundError:
    from scripts._bootstrap import REPO_ROOT, configure_runtime

configure_runtime()

from unml.config import (
    load_runtime_config,
    resolve_data_dir,
    resolve_model_value,
    resolve_section_value,
    resolve_value,
)
from unml.methods import UNLEARNING_METHODS
from unml.request_factory import (
    SelectiveRequest,
    build_selective_split,
    resolve_selective_request,
)

JOB_METHODS = UNLEARNING_METHODS
DEFAULT_METHOD = "ga_kl"
DEFAULT_STEPS = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one single-class CIFAR-100 unlearning job"
    )
    parser.add_argument(
        "--forget-class",
        required=True,
        help="CIFAR-100 class id or name (e.g. 70 or rose)",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=JOB_METHODS,
        help="Unlearning method",
    )
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS, help="Optimizer steps"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "config" / "parameters.yaml")
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--baseline-checkpoint",
        default=None,
        help="Class-agnostic fine-tuned checkpoint (required for a real run)",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split-path", default=None)
    parser.add_argument(
        "--rebuild-split",
        action="store_true",
        help="Recreate the split even if it already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved job plan without touching data or models",
    )
    return parser.parse_args()


def build_job_unlearn_config(
    runtime_cfg: dict,
    *,
    request: SelectiveRequest,
    dataset_name: str,
    data_dir: str,
    split_path: str,
    finetuned_checkpoint: str,
    output_dir: str,
    method: str,
    steps: int,
    seed: int,
    device: str,
    batch_size: int | None = None,
    num_workers: int | None = None,
):
    from unml.unlearn import UnlearnConfig

    return UnlearnConfig(
        data_dir=data_dir,
        split_path=split_path,
        finetuned_checkpoint=finetuned_checkpoint,
        output_dir=output_dir,
        method=method,
        dataset_name=dataset_name,
        model_name=resolve_model_value(
            None,
            runtime_cfg,
            dataset_name,
            "model_name",
            "openai/clip-vit-base-patch16",
        ),
        prompt_template=resolve_model_value(
            None,
            runtime_cfg,
            dataset_name,
            "prompt_template",
            "a photo of a {}",
        ),
        batch_size=resolve_value(
            batch_size, runtime_cfg, ("unlearning", "batch_size"), 128
        ),
        num_workers=resolve_section_value(
            num_workers,
            runtime_cfg,
            dataset_name,
            "unlearning",
            "num_workers",
            4,
        ),
        pin_memory=resolve_section_value(
            None, runtime_cfg, dataset_name, "unlearning", "pin_memory", True
        ),
        persistent_workers=resolve_section_value(
            None,
            runtime_cfg,
            dataset_name,
            "unlearning",
            "persistent_workers",
            False,
        ),
        prefetch_factor=resolve_section_value(
            None,
            runtime_cfg,
            dataset_name,
            "unlearning",
            "prefetch_factor",
            2,
        ),
        non_blocking=resolve_section_value(
            None,
            runtime_cfg,
            dataset_name,
            "unlearning",
            "non_blocking",
            False,
        ),
        steps=steps,
        lr=resolve_value(None, runtime_cfg, ("unlearning", "lr"), 5e-4),
        weight_decay=resolve_value(
            None, runtime_cfg, ("unlearning", "weight_decay"), 0.0
        ),
        seed=seed,
        device=device,
        kl_temperature=resolve_value(
            None, runtime_cfg, ("unlearning", "kl_temperature"), 1.5
        ),
        kl_weight=resolve_value(None, runtime_cfg, ("unlearning", "kl_weight"), 1.0),
        ga_weight=resolve_value(None, runtime_cfg, ("unlearning", "ga_weight"), 1.0),
        cf_weight=resolve_value(None, runtime_cfg, ("unlearning", "cf_weight"), 1.0),
        margin_weight=resolve_value(
            None, runtime_cfg, ("unlearning", "margin_weight"), 0.5
        ),
        margin=resolve_value(None, runtime_cfg, ("unlearning", "margin"), 0.2),
        entropy_weight=resolve_value(
            None, runtime_cfg, ("unlearning", "entropy_weight"), 1.0
        ),
        h_tgsd_target_weight=resolve_value(
            None,
            runtime_cfg,
            ("unlearning", "h_tgsd", "target_weight"),
            1.0,
        ),
        h_tgsd_entropy_weight=resolve_value(
            None,
            runtime_cfg,
            ("unlearning", "h_tgsd", "entropy_weight"),
            1.0,
        ),
        h_tgsd_unrelated_kl_weight=resolve_value(
            None,
            runtime_cfg,
            ("unlearning", "h_tgsd", "unrelated_kl_weight"),
            1.0,
        ),
        h_tgsd_unrelated_feature_weight=resolve_value(
            None,
            runtime_cfg,
            ("unlearning", "h_tgsd", "unrelated_feature_weight"),
            1.0,
        ),
        h_tgsd_sibling_kl_weight=resolve_value(
            None,
            runtime_cfg,
            ("unlearning", "h_tgsd", "sibling_kl_weight"),
            1.0,
        ),
        h_tgsd_sibling_feature_weight=resolve_value(
            None,
            runtime_cfg,
            ("unlearning", "h_tgsd", "sibling_feature_weight"),
            1.0,
        ),
        h_tgsd_shared_weight=resolve_value(
            None, runtime_cfg, ("unlearning", "h_tgsd", "shared_weight"), 1.0
        ),
        h_tgsd_text_weight=resolve_value(
            None, runtime_cfg, ("unlearning", "h_tgsd", "text_weight"), 0.5
        ),
        h_tgsd_prototype_samples_per_class=resolve_value(
            None,
            runtime_cfg,
            ("unlearning", "h_tgsd", "prototype_samples_per_class"),
            200,
        ),
        h_tgsd_basis_tolerance=resolve_value(
            None,
            runtime_cfg,
            ("unlearning", "h_tgsd", "basis_tolerance"),
            1e-5,
        ),
        train_logit_scale=resolve_model_value(
            None,
            runtime_cfg,
            dataset_name,
            "train_logit_scale_during_unlearning",
            True,
        ),
    )


def main() -> None:
    args = parse_args()
    runtime_cfg = load_runtime_config(args.config)
    dataset_name = "cifar100"

    request = resolve_selective_request(dataset_name, args.forget_class)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT
        / "outputs"
        / dataset_name
        / "jobs"
        / f"{request.request_name}_{args.method}_{args.steps}"
    )
    split_path = (
        Path(args.split_path)
        if args.split_path
        else output_dir / "splits" / f"{request.request_name}_split.json"
    )
    seed = (
        args.seed
        if args.seed is not None
        else int(runtime_cfg.get("experiment", {}).get("seed", 42))
    )

    plan = {
        "class_id": request.class_id,
        "class_name": request.class_name,
        "superclass": request.superclass,
        "sibling_classes": list(request.sibling_classes),
        "request_name": request.request_name,
        "method": args.method,
        "steps": args.steps,
        "seed": seed,
        "device": args.device,
        "output_dir": str(output_dir),
        "split_path": str(split_path),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    if not args.baseline_checkpoint:
        raise SystemExit(
            "--baseline-checkpoint is required for a real run "
            "(use --dry-run to inspect the plan only)"
        )

    data_dir = str(resolve_data_dir(args.data_dir, runtime_cfg))
    plan["data_dir"] = data_dir
    if split_path.is_file() and not args.rebuild_split:
        print(f"[job] reusing existing split {split_path}")
    else:
        build_selective_split(
            data_dir=data_dir,
            split_path=str(split_path),
            request=request,
            forget_fraction=1.0,
            retain_val_fraction=float(
                runtime_cfg.get("data", {}).get("retain_val_fraction", 0.1)
            ),
            seed=seed,
            dataset_name=dataset_name,
        )
        print(f"[job] wrote split {split_path}")

    from unml.unlearn import run_unlearning

    cfg = build_job_unlearn_config(
        runtime_cfg,
        request=request,
        dataset_name=dataset_name,
        data_dir=data_dir,
        split_path=str(split_path),
        finetuned_checkpoint=args.baseline_checkpoint,
        output_dir=str(output_dir),
        method=args.method,
        steps=args.steps,
        seed=seed,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    started = time.time()
    result = run_unlearning(cfg)
    wall_time = time.time() - started

    job_result = {
        **plan,
        "data_dir": data_dir,
        "finetuned_checkpoint": args.baseline_checkpoint,
        "wall_time_s": wall_time,
        "result": result,
    }
    result_path = output_dir / "job_result.json"
    result_path.write_text(json.dumps(job_result, indent=2), encoding="utf-8")
    print(json.dumps(job_result, indent=2))


if __name__ == "__main__":
    main()
