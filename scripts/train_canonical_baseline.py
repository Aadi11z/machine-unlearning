#!/usr/bin/env python3
"""Run one isolated canonical CIFAR-100 development training run."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from _bootstrap import REPO_ROOT, configure_runtime
except ModuleNotFoundError:
    from scripts._bootstrap import REPO_ROOT, configure_runtime

configure_runtime()

from unml.baseline import resolve_baseline_paths  # noqa: E402
from unml.canonical_split import load_canonical_cifar100_split  # noqa: E402
from unml.config import load_runtime_config  # noqa: E402
from unml.manifest import build_baseline_manifest, write_baseline_manifest  # noqa: E402
from unml.model import read_checkpoint_payload  # noqa: E402
from unml.train import FineTuneConfig, run_finetuning  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a canonical CIFAR-100 development run using the fixed "
            "18-template prompt contract and an isolated run directory."
        )
    )
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "outputs"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "canonical_cifar100.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument(
        "--final-fit",
        action="store_true",
        help="Train on all 50,000 official training examples, evaluate test, and write the immutable baseline manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = load_canonical_cifar100_split(args.split_path)
    config = load_runtime_config(str(args.config))
    values = config.get("canonical", {})
    if not isinstance(values, dict):
        raise ValueError("canonical config section must be a mapping")

    paths = resolve_baseline_paths(args.output_root, run_id=args.run_id)
    output_dir = paths.final_fit if args.final_fit else paths.development
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_existing:
        raise FileExistsError(
            f"Canonical run directory already contains artifacts: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    def value(name: str, fallback):
        return values.get(name, fallback)

    cfg = FineTuneConfig(
        data_dir=str(args.data_dir),
        split_path=str(args.split_path),
        output_dir=str(output_dir),
        dataset_name="cifar100",
        model_name=value("model_name", "openai/clip-vit-base-patch16"),
        adapter_type="vision_lora",
        lora_rank=int(value("lora_rank", 8)),
        lora_alpha=float(value("lora_alpha", 8.0)),
        lora_layers=value("lora_layers", "all"),
        lora_targets=value("lora_targets", ["q_proj", "v_proj"]),
        train_logit_scale=False,
        precision=args.precision or value("precision", "bf16"),
        gradient_checkpointing=bool(value("gradient_checkpointing", True)),
        batch_size=int(value("batch_size", 64)),
        gradient_accumulation_steps=int(value("gradient_accumulation_steps", 2)),
        num_workers=int(value("num_workers", 4)),
        pin_memory=bool(value("pin_memory", True)),
        persistent_workers=bool(value("persistent_workers", True)),
        prefetch_factor=int(value("prefetch_factor", 2)),
        non_blocking=bool(value("non_blocking", True)),
        lr=float(value("lr", 3e-3)),
        weight_decay=float(value("weight_decay", 1e-4)),
        epochs=args.epochs if args.epochs is not None else int(value("epochs", 10)),
        max_train_steps=(
            args.max_train_steps
            if args.max_train_steps is not None
            else int(value("max_train_steps", -1))
        ),
        seed=int(value("seed", 42)),
        device=args.device or value("device", "auto"),
        local_files_only=args.offline,
        evaluate_test=args.final_fit,
        canonical_final_fit=args.final_fit,
    )
    if cfg.seed != 42:
        raise ValueError("Canonical CIFAR-100 training is fixed to seed 42")
    print(f"[canonical-train] split={split['split_id']} digest={split['digest']}")
    print(f"[canonical-train] run_id={args.run_id} output={output_dir}")
    result = run_finetuning(cfg)
    if args.final_fit:
        checkpoint_path = Path(str(result["best_checkpoint"]))
        payload = read_checkpoint_payload(checkpoint_path)
        metrics_path = Path(str(result["metrics_path"]))
        import json

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest = build_baseline_manifest(
            baseline_id=args.run_id,
            dataset="cifar100",
            split=split,
            model_config=payload["model_config"],
            prompt_contract=payload["extra"]["prompt_contract"],
            checkpoints={"checkpoint": checkpoint_path},
            metrics=metrics,
            artifact_root=output_dir,
        )
        manifest_path = output_dir / "manifest.json"
        write_baseline_manifest(manifest_path, manifest)
        print(f"[canonical-train] manifest={manifest_path}")
    print(result)


if __name__ == "__main__":
    main()
