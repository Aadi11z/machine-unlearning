#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from unml.config import load_runtime_config, resolve_str_list, resolve_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one isolated study cell or build its result manifest"
    )
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "config" / "parameters.yaml")
    )
    parser.add_argument("--request", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--show-commands", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def dataset_root(runtime_cfg: dict, dataset: str) -> Path:
    environment_root = os.environ.get("UNML_OUTPUTS")
    if environment_root:
        return Path(environment_root) / dataset
    raw = str(runtime_cfg.get("outputs", {}).get("root", "outputs/{dataset}"))
    return Path(raw.format(dataset=dataset))


def cell_paths(root: Path, request: str, seed: int) -> dict[str, Path]:
    cell_root = root / request / f"seed_{seed}"
    return {
        "root": cell_root,
        "split": cell_root / "splits" / f"{request}_split.json",
        "comparison": cell_root / "comparison" / "comparison.csv",
    }


def build_pipeline_command(
    *,
    config: str,
    dataset: str,
    request: str,
    seed: int,
    device: str,
    root: Path,
) -> list[str]:
    paths = cell_paths(root, request, seed)
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_pipeline.py"),
        "--config",
        config,
        "--dataset",
        dataset,
        "--request",
        request,
        "--seed",
        str(seed),
        "--split-path",
        str(paths["split"]),
        "--output-root",
        str(paths["root"]),
        "--device",
        device,
    ]


def write_manifest(
    root: Path,
    requests: list[str],
    seeds: list[int],
    destination: Path,
) -> None:
    rows = []
    missing = []
    for request in requests:
        for seed in seeds:
            comparison = cell_paths(root, request, seed)["comparison"]
            if not comparison.is_file():
                missing.append(str(comparison))
            rows.append(
                {
                    "request": request,
                    "seed": seed,
                    "comparison_path": os.path.relpath(
                        comparison, destination.parent
                    ),
                }
            )
    if missing:
        raise FileNotFoundError(
            "Cannot write a complete study manifest; missing comparisons: "
            + ", ".join(missing)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(f"[study] manifest={destination}", flush=True)


def main() -> None:
    args = parse_args()
    runtime_cfg = load_runtime_config(args.config)
    dataset = str(
        resolve_value(
            None, runtime_cfg, ("study", "dataset"), "cifar100"
        )
    )
    requests = resolve_str_list(
        None,
        runtime_cfg,
        ("study", "requests"),
        ("flowers_superclass", "rose_selective"),
    )
    seeds = [
        int(seed)
        for seed in resolve_str_list(
            None,
            runtime_cfg,
            ("study", "seeds"),
            ("42", "123", "456"),
        )
    ]
    device = str(
        resolve_value(
            args.device, runtime_cfg, ("experiment", "device"), "auto"
        )
    )
    root = dataset_root(runtime_cfg, dataset)
    manifest = root / "study_manifest.csv"

    if args.write_manifest:
        write_manifest(root, requests, seeds, manifest)
        return
    if args.show_commands:
        for request in requests:
            for seed in seeds:
                command = build_pipeline_command(
                    config=args.config,
                    dataset=dataset,
                    request=request,
                    seed=seed,
                    device=device,
                    root=root,
                )
                print("[study]", " ".join(command), flush=True)
        return
    if args.request not in requests:
        raise ValueError(f"--request must be one of {requests}")
    if args.seed not in seeds:
        raise ValueError(f"--seed must be one of {seeds}")
    command = build_pipeline_command(
        config=args.config,
        dataset=dataset,
        request=args.request,
        seed=args.seed,
        device=device,
        root=root,
    )
    print("[study]", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=os.environ.copy())


if __name__ == "__main__":
    main()
