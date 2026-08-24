#!/usr/bin/env python3
"""Run the bounded Phase One LoRA schema screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from unml.manifest import sha256_file
from unml.prompts import resolve_prompt_contract
from unml.pilots import schema_screen_specs, build_pilot_report, write_pilot_report
from unml.train import FineTuneConfig, run_finetuning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    results: list[dict[str, object]] = []
    for spec in schema_screen_specs(seed=args.seed, steps=args.steps):
        run_dir = output_root / spec.name
        config = FineTuneConfig(
            data_dir=args.data_dir,
            split_path=args.split_path,
            output_dir=str(run_dir),
            dataset_name="cifar100",
            model_name=args.model_name,
            adapter_type="vision_lora",
            lora_rank=spec.lora_rank,
            lora_alpha=spec.lora_alpha,
            lora_targets=list(spec.lora_targets),
            lora_dropout=spec.lora_dropout,
            train_logit_scale=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=spec.lr,
            weight_decay=spec.weight_decay,
            max_train_steps=spec.max_train_steps,
            seed=spec.seed,
            device=args.device,
            local_files_only=args.local_files_only,
            evaluate_test=False,
            gradient_checkpointing=True,
        )
        print(f"[pilot] {spec.name}")
        result = run_finetuning(config)
        metrics = json.loads(Path(str(result["metrics_path"])).read_text())
        checkpoint = Path(str(result["best_checkpoint"]))
        architecture = metrics.get("architecture", {})
        results.append(
            {
                "name": spec.name,
                "stage": spec.stage,
                "seed": spec.seed,
                "config": spec.to_config(),
                "metrics": metrics["final_metrics"],
                "adapter_size_bytes": checkpoint.stat().st_size,
                "trainable_parameter_count": architecture.get("trainable_parameter_count"),
                "checkpoint": str(checkpoint),
            }
        )

    contract = resolve_prompt_contract("cifar100")
    report = build_pilot_report(
        split_path=args.split_path,
        split_digest=sha256_file(args.split_path),
        prompt_contract={"version": contract.version, "digest": contract.digest, "template_count": len(contract.templates)},
        results=results,
    )
    write_pilot_report(args.report_path, report)
    print(f"[pilot] report saved to {args.report_path}")


if __name__ == "__main__":
    main()
