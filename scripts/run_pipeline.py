#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from _bootstrap import REPO_ROOT, configure_runtime
except ModuleNotFoundError:
    from scripts._bootstrap import REPO_ROOT, configure_runtime

configure_runtime()

from unml.config import (
    load_runtime_config,
    resolve_data_dir,
    resolve_dataset_and_split_path,
    resolve_dataset_value,
    resolve_forget_classes,
    resolve_model_value,
    resolve_request_name,
    resolve_run_paths,
    resolve_section_str_list,
    resolve_section_value,
    resolve_value,
)
from unml.methods import H_TGSD_METHODS
from unml.pipeline import run_pipeline_stage


@dataclass(frozen=True)
class StageSpec:
    name: str
    command: list[str]
    inputs: list[str | Path]
    outputs: list[str | Path]


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip stages only when their command, Git/config/input hashes, "
            "and output hashes match a successful stage receipt"
        ),
    )
    parser.add_argument(
        "--force-stage",
        action="append",
        default=[],
        help=(
            "Rerun a stage even under --resume. Repeat as needed; valid names "
            "include prepare, finetune, retrain_oracle, unlearn, "
            "unlearn:<method>, and evaluate"
        ),
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
    retraining_enabled = bool(
        resolve_section_value(
            None,
            runtime_cfg,
            dataset_name,
            "retraining",
            "enabled",
            False,
        )
    )
    seed = resolve_value(args.seed, runtime_cfg, ("experiment", "seed"), 42)
    device = resolve_value(args.device, runtime_cfg, ("experiment", "device"), "auto")

    paths = resolve_run_paths(
        args.output_root, runtime_cfg, dataset_name, args.request
    )
    output_root = paths.output_root
    finetune_dir = paths.finetune_dir
    unlearn_dir = paths.unlearning_dir
    compare_dir = paths.comparison_dir
    config_path = Path(args.config).resolve()

    def run_stage(stage: StageSpec) -> None:
        force = stage.name in args.force_stage or (
            stage.name.startswith("unlearn:") and "unlearn" in args.force_stage
        )
        run_pipeline_stage(
            stage=stage.name,
            command=stage.command,
            env=env,
            output_root=output_root,
            input_paths=[config_path, *stage.inputs],
            output_paths=stage.outputs,
            resume=args.resume,
            force=force,
            repo_root=repo_root,
        )

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
        print(f"retraining.enabled={retraining_enabled}")
        if retraining_enabled:
            print(f"retraining_dir={output_root / 'retrain_oracle'}")
        print(f"training.num_workers={ft_num_workers}")
        print(f"training.pin_memory={ft_pin_memory}")
        print(f"training.persistent_workers={ft_persistent_workers}")
        print(f"training.prefetch_factor={ft_prefetch_factor}")
        print(f"training.non_blocking={ft_non_blocking}")
        print(f"seed={seed}")
        print(f"device={device}")
        print(f"resume={args.resume}")
        print(f"force_stages={','.join(args.force_stage)}")
        return

    request_args = ["--request", request_name] if request_name else []

    # Data Preparation command
    prepare_command = [
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
    ]
    run_stage(StageSpec("prepare", prepare_command, [], [split_path]))

    # Training VLM command
    finetune_command = [
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
    ]
    recovery_checkpoint = finetune_dir / "checkpoints" / "recovery_latest.pt"
    finetune_inputs: list[str | Path] = [split_path]
    if args.resume and recovery_checkpoint.is_file():
        finetune_command.extend(["--resume-checkpoint", str(recovery_checkpoint)])
        finetune_inputs.append(recovery_checkpoint)

    finetuned_ckpt = finetune_dir / "checkpoints" / "finetuned_best.pt"
    base_ckpt = finetune_dir / "checkpoints" / "base_init.pt"
    finetune_metrics = finetune_dir / "metrics" / "finetune_metrics.json"
    run_stage(
        StageSpec(
            "finetune",
            finetune_command,
            finetune_inputs,
            [base_ckpt, finetuned_ckpt, finetune_metrics],
        )
    )

    oracle_dir = paths.oracle_dir
    oracle_ckpt = oracle_dir / "checkpoints" / "retrained_best.pt"
    oracle_metrics = oracle_dir / "metrics" / "retrain_metrics.json"

    if retraining_enabled:
        oracle_command = [
                sys.executable,
                str(repo_root / "scripts" / "train_vlm.py"),
                "--config",
                args.config,
                "--oracle",
                "--data-dir",
                data_dir,
                "--dataset",
                dataset_name,
                *request_args,
                "--split-path",
                split_path,
                "--output-dir",
                str(oracle_dir),
                "--initial-checkpoint",
                str(base_ckpt),
                "--source-metrics",
                str(finetune_metrics),
                "--model-name",
                model_name,
                "--batch-size",
                str(ft_batch_size),
                "--gradient-accumulation-steps",
                str(ft_gradient_accumulation_steps),
                "--num-workers",
                str(ft_num_workers),
                "--seed",
                str(seed),
                "--device",
                device,
        ]
        run_stage(
            StageSpec(
                "retrain_oracle",
                oracle_command,
                [split_path, base_ckpt, finetune_metrics],
                [oracle_ckpt, oracle_metrics],
            )
        )

    for method in methods:
        unlearn_ckpt = (
            unlearn_dir / "checkpoints" / f"unlearn_{method}.pt"
        )
        unlearn_metrics = (
            unlearn_dir / "metrics" / f"unlearn_{method}_metrics.json"
        )
        unlearn_outputs: list[str | Path] = [
            unlearn_ckpt,
            unlearn_metrics,
        ]
        if method in H_TGSD_METHODS:
            unlearn_outputs.append(
                unlearn_dir / "metrics" / f"{method}_basis.pt"
            )
        unlearn_command = [
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
        ]
        run_stage(
            StageSpec(
                f"unlearn:{method}",
                unlearn_command,
                [split_path, finetuned_ckpt],
                unlearn_outputs,
            )
        )

    candidate_args = [
        "--candidate",
        f"finetuned={finetuned_ckpt}",
    ]
    if retraining_enabled:
        candidate_args.extend(
            ["--candidate", f"retrain_oracle={oracle_ckpt}"]
        )
    for method in methods:
        candidate_args.extend(
            [
                "--candidate",
                f"{method}={unlearn_dir / 'checkpoints' / f'unlearn_{method}.pt'}",
            ]
        )

    evaluate_command = [
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
    ]
    evaluation_inputs: list[str | Path] = [
        split_path,
        base_ckpt,
        finetuned_ckpt,
    ]
    if retraining_enabled:
        evaluation_inputs.append(oracle_ckpt)
    evaluation_inputs.extend(
        unlearn_dir / "checkpoints" / f"unlearn_{method}.pt"
        for method in methods
    )
    evaluation_outputs: list[str | Path] = [
        compare_dir / "comparison.csv",
        compare_dir / "comparison.md",
        compare_dir / "behavioral_tradeoff.png",
        compare_dir / "plot_manifest.json",
    ]
    if any(method in H_TGSD_METHODS for method in methods):
        evaluation_outputs.append(compare_dir / "semantic_subspace.png")
    run_stage(
        StageSpec(
            "evaluate",
            evaluate_command,
            evaluation_inputs,
            evaluation_outputs,
        )
    )

    print(f"[done] comparison markdown: {compare_dir / 'comparison.md'}")
    print(
        f"[done] behavioral tradeoff: "
        f"{compare_dir / 'behavioral_tradeoff.png'}"
    )
    print(
        f"[done] semantic subspace: "
        f"{compare_dir / 'semantic_subspace.png'}"
    )
    print(f"[done] plot manifest: {compare_dir / 'plot_manifest.json'}")


if __name__ == "__main__":
    main()
