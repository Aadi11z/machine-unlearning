#!/usr/bin/env python3
"""Hyperparameter sweep for finetuning.

Iterates over a grid of (adapter_rank, adapter_alpha, lr, weight_decay, seed)
and runs finetuning for each combination. Results are ranked by retain_val_acc
and the best checkpoint path is printed at the end.

Data only needs to be prepared once (prepare_data.py); this script reuses
the existing data dir and split file across all runs.

Usage:
    python scripts/sweep_finetune.py --data-dir data --split-path outputs/splits/cifar10_split.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep finetuning hyperparameters")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--split-path", type=str, default="outputs/splits/cifar10_split.json")
    p.add_argument("--output-root", type=str, default="outputs/sweep_finetune",
                    help="Root dir; each run gets a subdirectory")
    p.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")

    # Sweep values (comma-separated)
    p.add_argument("--ranks", type=str, default="8,16,32",
                    help="Comma-separated adapter_rank values")
    p.add_argument("--alphas", type=str, default="8.0,16.0,32.0",
                    help="Comma-separated adapter_alpha values")
    p.add_argument("--lrs", type=str, default="5e-4,1e-3,3e-3",
                    help="Comma-separated learning rates")
    p.add_argument("--weight-decays", type=str, default="1e-4,1e-3",
                    help="Comma-separated weight_decay values")
    p.add_argument("--seeds", type=str, default="42",
                    help="Comma-separated seeds")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ranks = [int(x) for x in args.ranks.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    lrs = [float(x) for x in args.lrs.split(",")]
    weight_decays = [float(x) for x in args.weight_decays.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    combos = list(itertools.product(ranks, alphas, lrs, weight_decays, seeds))
    total = len(combos)
    print(f"[sweep] {total} configurations to run\n")

    from unml.train import FineTuneConfig, run_finetuning

    results: list[dict] = []
    output_root = Path(args.output_root)

    for i, (rank, alpha, lr, wd, seed) in enumerate(combos, 1):
        run_name = f"r{rank}_a{alpha}_lr{lr}_wd{wd}_s{seed}"
        run_dir = output_root / run_name
        print(f"\n{'='*60}")
        print(f"[sweep] Run {i}/{total}: {run_name}")
        print(f"{'='*60}")

        cfg = FineTuneConfig(
            data_dir=args.data_dir,
            split_path=args.split_path,
            output_dir=str(run_dir),
            model_name=args.model_name,
            adapter_rank=rank,
            adapter_alpha=alpha,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=lr,
            weight_decay=wd,
            epochs=args.epochs,
            seed=seed,
            device=args.device,
        )

        try:
            result = run_finetuning(cfg)
            entry = {
                "run_name": run_name,
                "adapter_rank": rank,
                "adapter_alpha": alpha,
                "lr": lr,
                "weight_decay": wd,
                "seed": seed,
                "best_retain_val_acc": result["best_retain_val_acc"],
                "best_checkpoint": result["best_checkpoint"],
            }
            results.append(entry)
            print(f"[sweep] {run_name} => retain_val_acc={result['best_retain_val_acc']:.4f}")
        except Exception as e:
            print(f"[sweep] {run_name} FAILED: {e}")
            results.append({"run_name": run_name, "best_retain_val_acc": -1, "error": str(e)})

    # Rank and save results
    results.sort(key=lambda r: r.get("best_retain_val_acc", -1), reverse=True)

    summary_path = output_root / "sweep_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("[sweep] RESULTS (ranked by retain_val_acc):")
    print(f"{'='*60}")
    for j, r in enumerate(results, 1):
        acc = r.get("best_retain_val_acc", -1)
        if acc < 0:
            print(f"  {j}. {r['run_name']} => FAILED")
        else:
            print(f"  {j}. {r['run_name']} => {acc:.4f}")

    if results and results[0].get("best_retain_val_acc", -1) > 0:
        best = results[0]
        print(f"\n[sweep] Best config: {best['run_name']}")
        print(f"[sweep] Best checkpoint: {best.get('best_checkpoint')}")
        print(f"[sweep] retain_val_acc: {best['best_retain_val_acc']:.4f}")

    print(f"\n[sweep] Full results saved to {summary_path}")


if __name__ == "__main__":
    main()


# usage
# cleanup_checkpoints.py — standalone cleanup tool with these rules:
#   - base_init.pt is always targeted (it's identical across all runs/methods)
#   - finetuned_last.pt is only targeted when you pass --include-last              
#   - finetuned_best.pt is never deleted       
#   - --run-name lets you scope --include-last to a specific config
#   - --include-oversized flags old 577 MB checkpoints that still have the frozen backbone                       
#   - Default is dry-run preview; --delete actually removes files                                                                         
#   # Preview what's redundant                                               
#   python scripts/cleanup_checkpoints.py                                                                    
#   # Delete all base_init.pt files                             
#   python scripts/cleanup_checkpoints.py --delete                                                 
#   # Delete last checkpoint for one specific run only                                                     
#   python scripts/cleanup_checkpoints.py --delete --include-last --run-name r8_a8.0_lr0.001_wd0.0001_s42                                             
#   # Nuclear option: also clean old oversized checkpoints                             
#   python scripts/cleanup_checkpoints.py --delete --include-oversized 