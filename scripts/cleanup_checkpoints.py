#!/usr/bin/env python3
"""Clean up redundant checkpoint files to free disk space.

base_init.pt is always redundant — it's the randomly initialized adapters
before any training, identical across all runs and methods. It can always
be recreated by re-instantiating the model.

finetuned_last.pt is the last-epoch snapshot. It may differ from
finetuned_best.pt, so it is only deleted when you explicitly target a
specific run config with --run-name and pass --include-last.

Usage:
    # Preview all redundant base_init.pt files
    python scripts/cleanup_checkpoints.py

    # Delete all base_init.pt files
    python scripts/cleanup_checkpoints.py --delete

    # Also delete finetuned_last.pt for a specific sweep run
    python scripts/cleanup_checkpoints.py --delete --include-last --run-name r8_a8.0_lr0.001_wd0.0001_s42

    # Delete finetuned_last.pt across all runs
    python scripts/cleanup_checkpoints.py --delete --include-last

    # Also flag old oversized checkpoints (>1 MB, contain frozen CLIP backbone)
    python scripts/cleanup_checkpoints.py --delete --include-oversized
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"

OVERSIZED_THRESHOLD = 1_000_000  # 1 MB — adapter-only checkpoints are ~100 KB


def _size_str(size_bytes: int) -> str:
    if size_bytes > 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / 1024:.1f} KB"


def find_targets(outputs_dir: Path, include_last: bool, include_oversized: bool, run_name: str | None) -> list[tuple[Path, str]]:
    """Return list of (path, reason) for files that can be deleted."""
    targets: list[tuple[Path, str]] = []

    for pt_file in sorted(outputs_dir.rglob("*.pt")):
        # If --run-name is specified, only target files under that run's directory
        if run_name and run_name not in str(pt_file):
            # base_init.pt is always redundant regardless of run_name filter
            if pt_file.name != "base_init.pt":
                continue

        if pt_file.name == "base_init.pt":
            targets.append((pt_file, "redundant (random init, always recreatable)"))
        elif pt_file.name == "finetuned_last.pt" and include_last:
            targets.append((pt_file, "last-epoch snapshot (best checkpoint retained)"))
        elif include_oversized:
            size_bytes = pt_file.stat().st_size
            if size_bytes > OVERSIZED_THRESHOLD:
                targets.append((pt_file, f"oversized ({_size_str(size_bytes)}, likely contains frozen CLIP backbone)"))

    return targets


def main() -> None:
    p = argparse.ArgumentParser(description="Clean up redundant checkpoint files")
    p.add_argument("--outputs-dir", type=str, default=str(OUTPUTS_DIR),
                    help="Root outputs directory to scan")
    p.add_argument("--delete", action="store_true",
                    help="Actually delete files (default is dry-run preview)")
    p.add_argument("--include-last", action="store_true",
                    help="Also target finetuned_last.pt files")
    p.add_argument("--include-oversized", action="store_true",
                    help="Also target .pt files >1 MB (old full-model checkpoints)")
    p.add_argument("--run-name", type=str, default=None,
                    help="Only target finetuned_last.pt under this specific run "
                         "(e.g. r8_a8.0_lr0.001_wd0.0001_s42). base_init.pt is "
                         "always targeted regardless.")
    args = p.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        print(f"[cleanup] Directory not found: {outputs_dir}")
        return

    targets = find_targets(outputs_dir, args.include_last, args.include_oversized, args.run_name)

    if not targets:
        print("[cleanup] No redundant checkpoints found.")
        return

    total_bytes = 0
    for path, reason in targets:
        size = path.stat().st_size
        total_bytes += size
        action = "DELETE" if args.delete else "WOULD DELETE"
        print(f"  [{action}] {path}  ({_size_str(size)}, {reason})")
        if args.delete:
            path.unlink()

    if args.delete:
        print(f"\n[cleanup] Deleted {len(targets)} files, freed {_size_str(total_bytes)}")
    else:
        print(f"\n[cleanup] Found {len(targets)} files ({_size_str(total_bytes)})")
        print("[cleanup] Run with --delete to remove them")


if __name__ == "__main__":
    main()
