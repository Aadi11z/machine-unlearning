#!/usr/bin/env python3
"""Write and verify provenance for one completed canonical retraining oracle."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import tempfile

try:
    from _bootstrap import configure_runtime
except ModuleNotFoundError:
    from scripts._bootstrap import configure_runtime

configure_runtime()

from unml.baseline import resolve_baseline  # noqa: E402
from unml.manifest import (  # noqa: E402
    build_retraining_oracle_manifest,
    sha256_file,
    validate_baseline_identity,
    verify_retraining_oracle_canonical_contract,
    verify_retraining_oracle_manifest,
    write_retraining_oracle_manifest,
)
from unml.model import read_checkpoint_payload  # noqa: E402
from unml.prompts import resolve_prompt_contract  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument(
        "--oracle-id", default="cifar100_retraining_oracle_v1"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    oracle_dir = args.oracle_dir.resolve()
    canonical_dir = args.canonical_dir.resolve()
    canonical_manifest_path = canonical_dir / "manifest.json"
    baseline = resolve_baseline(canonical_manifest_path)
    canonical = baseline.manifest
    canonical_checkpoint = baseline.final_checkpoint
    validate_baseline_identity(
        canonical,
        baseline_id=baseline.baseline_id,
        checkpoint_path=canonical_checkpoint,
    )

    checkpoint = oracle_dir / "checkpoints" / "retrained_best.pt"
    metrics_path = oracle_dir / "metrics" / "retrain_metrics.json"
    comparison_path = oracle_dir / "metrics" / "reference_comparison.json"
    split_candidates = sorted((oracle_dir / "splits").glob("*.json"))
    if len(split_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one oracle split in {oracle_dir / 'splits'}, "
            f"found {len(split_candidates)}"
        )
    split_path = split_candidates[0]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    payload = read_checkpoint_payload(checkpoint)
    contract = resolve_prompt_contract("cifar100")
    stored_contract = payload.get("extra", {}).get("prompt_contract", {})
    if stored_contract.get("digest") != contract.digest:
        raise ValueError("Oracle checkpoint does not use the canonical prompt contract")
    if metrics.get("training_mode") != "retrain_oracle":
        raise ValueError("Oracle metrics do not identify retrain_oracle training")

    forget_classes = [int(value) for value in split["forget_classes"]]
    class_names = list(split["class_names"])
    request = {
        "request_name": split["request_name"],
        "request_type": split["request_type"],
        "forget_classes": forget_classes,
        "forget_class_names": [class_names[index] for index in forget_classes],
        "test_forget_count": len(split["test_forget_indices"]),
        "test_retain_count": len(split["test_retain_indices"]),
    }
    manifest = build_retraining_oracle_manifest(
        oracle_id=args.oracle_id,
        dataset="cifar100",
        request=request,
        canonical_contract={
            "baseline_id": canonical["baseline_id"],
            "manifest_sha256": sha256_file(canonical_manifest_path),
            "prompt_digest": canonical["prompt_contract"]["digest"],
            "split_id": canonical["split"]["split_id"],
            "split_digest": canonical["split"]["digest"],
        },
        model_config=payload["model_config"],
        prompt_contract=stored_contract,
        checkpoint=checkpoint,
        metrics_path=metrics_path,
        split_path=split_path,
        comparison_path=comparison_path,
        metrics=metrics,
        comparison=comparison,
        artifact_root=oracle_dir,
    )
    verify_retraining_oracle_manifest(manifest, root=oracle_dir)
    verify_retraining_oracle_canonical_contract(
        manifest,
        canonical,
        canonical_manifest_path=canonical_manifest_path,
    )
    destination = oracle_dir / "manifest.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != manifest:
            raise FileExistsError(
                f"Refusing to replace immutable oracle manifest: {destination}"
            )
    else:
        write_retraining_oracle_manifest(destination, manifest)
    _promote_reference(oracle_dir, destination)
    print(f"[oracle-manifest] verified={destination}")


def _promote_reference(oracle_dir: Path, manifest_path: Path) -> None:
    references_root = oracle_dir.parents[1]
    request_name = oracle_dir.name
    index_path = references_root / "promoted.json"
    lock_path = references_root / ".promoted.lock"
    references_root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
        else:
            index = {
                "schema": "unml-retraining-oracle-index-v1",
                "references": {},
            }
        if index.get("schema") != "unml-retraining-oracle-index-v1" or not isinstance(
            index.get("references"), dict
        ):
            raise ValueError(f"Invalid retraining oracle index: {index_path}")
        index["references"][request_name] = str(
            manifest_path.relative_to(references_root)
        )
        fd, temporary_name = tempfile.mkstemp(
            prefix=".promoted.", suffix=".tmp", dir=references_root
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(index, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, index_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


if __name__ == "__main__":
    main()
