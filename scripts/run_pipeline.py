#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full lightweight VLM unlearning pipeline")

    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split-path", type=str, default="outputs/splits/cifar10_split.json")
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--forget-classes", type=str, default="3,5")
    parser.add_argument("--forget-fraction", type=float, default=1.0)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ft-epochs", type=int, default=5)
    parser.add_argument("--ft-max-steps", type=int, default=-1)
    parser.add_argument("--ul-steps", type=int, default=500)
    parser.add_argument("--methods", type=str, default="retain_only,ga_kl,counterfactual_rebind")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1] # `resolve` converts it to absolute path, and parents[1] goes 2 levels up
    env = os.environ.copy() # copies current shell environment variables
    env.setdefault("TRANSFORMERS_NO_TF", "1") # Tell HuggingFace Transformers, not to use TensorFlow and Flax to prevent slow imports, dependency confilcts, 
    env.setdefault("TRANSFORMERS_NO_FLAX", "1") # and CUDA issues
    env.setdefault("USE_TF", "0")
    env.setdefault("USE_FLAX", "0")
    env.setdefault("TOKENIZERS_PARALLELISM", "false") # Prevents HuggingFace Transformers from spawning many threads, and to avoid deadlock situations
    env.setdefault("PYTHONUNBUFFERED", "1") # for logs to appear immediately
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    # Defining output directories
    finetune_dir = Path(args.output_root) / "finetune"
    unlearn_dir = Path(args.output_root) / "unlearning"
    compare_dir = Path(args.output_root) / "comparison"

    # Data Preparation command
    run_cmd(
        [
            sys.executable,
            str(repo_root / "scripts" / "prepare_data.py"),
            "--data-dir",
            args.data_dir,
            "--split-path",
            args.split_path,
            "--forget-classes",
            args.forget_classes,
            "--forget-fraction",
            str(args.forget_fraction),
            "--seed",
            str(args.seed),
        ],
        env,
    )

    # Training VLM command
    run_cmd(
        [
            sys.executable,
            str(repo_root / "scripts" / "train_vlm.py"),
            "--data-dir",
            args.data_dir,
            "--split-path",
            args.split_path,
            "--output-dir",
            str(finetune_dir),
            "--model-name",
            args.model_name,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--epochs",
            str(args.ft_epochs),
            "--max-train-steps",
            str(args.ft_max_steps),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ],
        env,
    )

    finetuned_ckpt = finetune_dir / "checkpoints" / "finetuned_best.pt"
    base_ckpt = finetune_dir / "checkpoints" / "base_init.pt"

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        run_cmd(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_unlearning.py"),
                "--data-dir",
                args.data_dir,
                "--split-path",
                args.split_path,
                "--finetuned-checkpoint",
                str(finetuned_ckpt),
                "--output-dir",
                str(unlearn_dir),
                "--method",
                method,
                "--model-name",
                args.model_name,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--steps",
                str(args.ul_steps),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
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
            "--data-dir",
            args.data_dir,
            "--split-path",
            args.split_path,
            "--model-name",
            args.model_name,
            "--base-checkpoint",
            str(base_ckpt),
            "--output-dir",
            str(compare_dir),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
            *candidate_args,
        ],
        env,
    )

    print(f"[done] comparison markdown: {compare_dir / 'comparison.md'}")
    print(f"[done] comparison plot: {compare_dir / 'utility_vs_forget.png'}")

if __name__ == "__main__":
    main()
