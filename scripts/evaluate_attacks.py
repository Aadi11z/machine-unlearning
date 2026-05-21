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

from unml.config import load_runtime_config, resolve_str_list, resolve_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run attack suite and compare utility/forget quality"
    )
    parser.add_argument(
        "--config", type=str, default=str(REPO_ROOT / "config" / "parameters.yaml")
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--split-path", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--base-checkpoint", type=str, default=None)

    parser.add_argument(
        "--candidate",
        type=str,
        action="append",
        help="Candidate checkpoint in the form name=path",
    )

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--prompt-template", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-attack-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_cfg = load_runtime_config(args.config)

    from unml.attacks import AttackConfig, run_attack_comparison

    names = []
    checkpoints = []
    if args.candidate:
        for entry in args.candidate:
            if "=" not in entry:
                raise ValueError(f"Invalid --candidate '{entry}'. Use name=path")
            name, path = entry.split("=", 1)
            names.append(name)
            checkpoints.append(path)
    else:
        training_output_dir = Path(
            resolve_value(
                None, runtime_cfg, ("training", "output_dir"), "outputs/finetune"
            )
        )
        unlearning_output_dir = Path(
            resolve_value(
                None, runtime_cfg, ("unlearning", "output_dir"), "outputs/unlearning"
            )
        )
        methods = resolve_str_list(
            None,
            runtime_cfg,
            ("unlearning", "methods"),
            ("retain_only", "ga_kl", "counterfactual_rebind"),
        )
        names.append("finetuned")
        checkpoints.append(
            str(training_output_dir / "checkpoints" / "finetuned_best.pt")
        )
        for method in methods:
            names.append(method)
            checkpoints.append(
                str(unlearning_output_dir / "checkpoints" / f"unlearn_{method}.pt")
            )

    if not checkpoints:
        raise ValueError("No candidates provided or derived for attack evaluation")

    cfg = AttackConfig(
        data_dir=resolve_value(
            args.data_dir, runtime_cfg, ("data", "data_dir"), "data"
        ),
        split_path=resolve_value(
            args.split_path,
            runtime_cfg,
            ("data", "split_path"),
            "outputs/splits/cifar10_split.json",
        ),
        model_name=resolve_value(
            args.model_name,
            runtime_cfg,
            ("model", "model_name"),
            "openai/clip-vit-base-patch32",
        ),
        base_checkpoint=resolve_value(
            args.base_checkpoint,
            runtime_cfg,
            ("attack", "base_checkpoint"),
            str(
                Path(
                    resolve_value(
                        None,
                        runtime_cfg,
                        ("training", "output_dir"),
                        "outputs/finetune",
                    )
                )
                / "checkpoints"
                / "base_init.pt"
            ),
        ),
        candidate_checkpoints=checkpoints,
        candidate_names=names,
        output_dir=resolve_value(
            args.output_dir, runtime_cfg, ("attack", "output_dir"), "outputs/comparison"
        ),
        prompt_template=resolve_value(
            args.prompt_template,
            runtime_cfg,
            ("model", "prompt_template"),
            "a photo of a {}",
        ),
        batch_size=resolve_value(
            args.batch_size, runtime_cfg, ("attack", "batch_size"), 128
        ),
        num_workers=resolve_value(
            args.num_workers, runtime_cfg, ("attack", "num_workers"), 4
        ),
        max_attack_samples=resolve_value(
            args.max_attack_samples, runtime_cfg, ("attack", "max_attack_samples"), 4000
        ),
        device=resolve_value(
            args.device, runtime_cfg, ("experiment", "device"), "auto"
        ),
    )

    result = run_attack_comparison(cfg)
    print(result)


if __name__ == "__main__":
    main()
