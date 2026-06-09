#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from unml.config import (
    load_runtime_config,
    resolve_data_dir,
    resolve_dataset_and_split_path,
    resolve_dataset_value,
    resolve_forget_classes,
    resolve_model_value,
    resolve_output_root,
    resolve_request_name,
    resolve_section_str_list,
    resolve_section_value,
    resolve_value,
)


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full lightweight VLM unlearning pipeline"
    )

    parser.add_argument(
        "--config", type=str, default=str(REPO_ROOT / "config" / "parameters.yaml")
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--request", type=str, default=None)
    parser.add_argument("--split-path", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--forget-classes", type=str, default=None)
    parser.add_argument("--forget-fraction", type=float, default=None)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--ft-epochs", type=int, default=None)
    parser.add_argument("--ft-max-steps", type=int, default=None)
    parser.add_argument("--ul-steps", type=int, default=None)
    parser.add_argument("--methods", type=str, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print resolved dataset and artifact paths without running stages",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_cfg = load_runtime_config(args.config)
    dataset_name, split_path = resolve_dataset_and_split_path(
        args.dataset, args.split_path, runtime_cfg, args.request
    )
    request_name = resolve_request_name(args.request, runtime_cfg, dataset_name)

    repo_root = REPO_ROOT
    env = os.environ.copy()
    env.setdefault("TRANSFORMERS_NO_TF", "1")
    env.setdefault("TRANSFORMERS_NO_FLAX", "1")
    env.setdefault("USE_TF", "0")
    env.setdefault("USE_FLAX", "0")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")  # for logs to appear immediately
    python_paths = [str(repo_root), str(repo_root / "src")]
    env["PYTHONPATH"] = os.pathsep.join(python_paths) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    data_dir = str(resolve_data_dir(args.data_dir, runtime_cfg))
    model_name = resolve_model_value(
        args.model_name,
        runtime_cfg,
        dataset_name,
        "model_name",
        "openai/clip-vit-base-patch32",
    )
    forget_classes = resolve_forget_classes(
        args.forget_classes, runtime_cfg, dataset_name, args.request
    )
    forget_fraction = resolve_dataset_value(
        args.forget_fraction,
        runtime_cfg,
        dataset_name,
        "forget_fraction",
        1.0,
    )
    ft_batch_size = resolve_section_value(
        args.batch_size,
        runtime_cfg,
        dataset_name,
        "training",
        "batch_size",
        128,
    )
    ft_num_workers = resolve_section_value(
        args.num_workers,
        runtime_cfg,
        dataset_name,
        "training",
        "num_workers",
        4,
    )
    ft_epochs = resolve_value(args.ft_epochs, runtime_cfg, ("training", "epochs"), 10)
    ft_max_steps = resolve_value(
        args.ft_max_steps, runtime_cfg, ("training", "max_train_steps"), -1
    )
    ft_gradient_accumulation_steps = resolve_section_value(
        None,
        runtime_cfg,
        dataset_name,
        "training",
        "gradient_accumulation_steps",
        1,
    )
    ul_batch_size = resolve_value(
        args.batch_size, runtime_cfg, ("unlearning", "batch_size"), 128
    )
    ul_num_workers = resolve_section_value(
        args.num_workers,
        runtime_cfg,
        dataset_name,
        "unlearning",
        "num_workers",
        4,
    )
    ul_steps = resolve_value(args.ul_steps, runtime_cfg, ("unlearning", "steps"), 500)
    attack_batch_size = resolve_value(
        args.batch_size, runtime_cfg, ("attack", "batch_size"), 128
    )
    attack_num_workers = resolve_section_value(
        args.num_workers,
        runtime_cfg,
        dataset_name,
        "attack",
        "num_workers",
        4,
    )
    ft_pin_memory = resolve_section_value(
        None, runtime_cfg, dataset_name, "training", "pin_memory", True
    )
    ft_persistent_workers = resolve_section_value(
        None, runtime_cfg, dataset_name, "training", "persistent_workers", False
    )
    ft_prefetch_factor = resolve_section_value(
        None, runtime_cfg, dataset_name, "training", "prefetch_factor", 2
    )
    ft_non_blocking = resolve_section_value(
        None, runtime_cfg, dataset_name, "training", "non_blocking", False
    )
    methods = resolve_section_str_list(
        args.methods,
        runtime_cfg,
        dataset_name,
        "unlearning",
        "methods",
        ("retain_only", "ga_kl", "counterfactual_rebind"),
    )
    seed = resolve_value(args.seed, runtime_cfg, ("experiment", "seed"), 42)
    device = resolve_value(args.device, runtime_cfg, ("experiment", "device"), "auto")

    output_root = resolve_output_root(
        args.output_root, runtime_cfg, dataset_name, args.request
    )
    finetune_dir = output_root / "finetune"
    unlearn_dir = output_root / "unlearning"
    compare_dir = output_root / "comparison"

    if args.show_config:
        print(f"dataset={dataset_name}")
        print(f"request={request_name or 'class_list'}")
        print(f"data_dir={data_dir}")
        print(f"split_path={split_path}")
        print(f"forget_classes={forget_classes}")
        print(f"forget_fraction={forget_fraction}")
        print(f"model_name={model_name}")
        print(
            "adapter_type="
            f"{resolve_model_value(None, runtime_cfg, dataset_name, 'adapter_type', 'post_projection')}"
        )
        print(f"output_root={output_root}")
        print(f"finetune_dir={finetune_dir}")
        print(f"unlearning_dir={unlearn_dir}")
        print(f"comparison_dir={compare_dir}")
        print(f"methods={','.join(methods)}")
        print(f"training.num_workers={ft_num_workers}")
        print(f"training.pin_memory={ft_pin_memory}")
        print(f"training.persistent_workers={ft_persistent_workers}")
        print(f"training.prefetch_factor={ft_prefetch_factor}")
        print(f"training.non_blocking={ft_non_blocking}")
        print(f"seed={seed}")
        print(f"device={device}")
        return

    request_args = ["--request", request_name] if request_name else []

    # Data Preparation command
    run_cmd(
        [
            sys.executable,
            str(repo_root / "scripts" / "prepare_data.py"),
            "--config",
            args.config,
            "--data-dir",
            data_dir,
            "--dataset",
            dataset_name,
            *request_args,
            "--split-path",
            split_path,
            "--forget-classes",
            forget_classes,
            "--forget-fraction",
            str(forget_fraction),
            "--seed",
            str(seed),
        ],
        env,
    )

    # Training VLM command
    run_cmd(
        [
            sys.executable,
            str(repo_root / "scripts" / "train_vlm.py"),
            "--config",
            args.config,
            "--data-dir",
            data_dir,
            "--dataset",
            dataset_name,
            *request_args,
            "--split-path",
            split_path,
            "--output-dir",
            str(finetune_dir),
            "--model-name",
            model_name,
            "--batch-size",
            str(ft_batch_size),
            "--gradient-accumulation-steps",
            str(ft_gradient_accumulation_steps),
            "--num-workers",
            str(ft_num_workers),
            "--epochs",
            str(ft_epochs),
            "--max-train-steps",
            str(ft_max_steps),
            "--seed",
            str(seed),
            "--device",
            device,
        ],
        env,
    )

    finetuned_ckpt = finetune_dir / "checkpoints" / "finetuned_best.pt"
    base_ckpt = finetune_dir / "checkpoints" / "base_init.pt"

    for method in methods:
        run_cmd(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_unlearning.py"),
                "--config",
                args.config,
                "--data-dir",
                data_dir,
                "--dataset",
                dataset_name,
                *request_args,
                "--split-path",
                split_path,
                "--finetuned-checkpoint",
                str(finetuned_ckpt),
                "--output-dir",
                str(unlearn_dir),
                "--method",
                method,
                "--model-name",
                model_name,
                "--batch-size",
                str(ul_batch_size),
                "--num-workers",
                str(ul_num_workers),
                "--steps",
                str(ul_steps),
                "--seed",
                str(seed),
                "--device",
                device,
            ],
            env,
        )

    candidate_args = [
        "--candidate",
        f"finetuned={finetuned_ckpt}",
    ]
    for method in methods:
        candidate_args.extend(
            [
                "--candidate",
                f"{method}={unlearn_dir / 'checkpoints' / f'unlearn_{method}.pt'}",
            ]
        )

    run_cmd(
        [
            sys.executable,
            str(repo_root / "scripts" / "evaluate_attacks.py"),
            "--config",
            args.config,
            "--data-dir",
            data_dir,
            "--dataset",
            dataset_name,
            *request_args,
            "--split-path",
            split_path,
            "--model-name",
            model_name,
            "--base-checkpoint",
            str(base_ckpt),
            "--output-dir",
            str(compare_dir),
            "--batch-size",
            str(attack_batch_size),
            "--num-workers",
            str(attack_num_workers),
            "--device",
            device,
            *candidate_args,
        ],
        env,
    )

    print(f"[done] comparison markdown: {compare_dir / 'comparison.md'}")
    print(f"[done] comparison plot: {compare_dir / 'utility_vs_forget.png'}")


if __name__ == "__main__":
    main()
