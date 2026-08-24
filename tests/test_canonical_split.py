from __future__ import annotations

import copy
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from unml.canonical_split import (
    CANONICAL_CIFAR100_SPLIT_ID,
    build_canonical_cifar100_split,
    canonical_training_payload,
    load_canonical_cifar100_split,
    validate_canonical_cifar100_split,
    validate_canonical_cifar100_dataset_labels,
    write_canonical_cifar100_split,
)
from unml.data import SplitConfig, make_splits


def _official_cifar100_labels(*, per_class: int) -> list[int]:
    return [class_id for class_id in range(100) for _ in range(per_class)]


@pytest.fixture
def canonical_payload() -> dict[str, object]:
    return build_canonical_cifar100_split(
        _official_cifar100_labels(per_class=500),
        _official_cifar100_labels(per_class=100),
    )


def test_canonical_split_is_stratified_complete_and_digest_verified(
    canonical_payload: dict[str, object],
) -> None:
    validate_canonical_cifar100_split(canonical_payload)

    train_labels = canonical_payload["train_labels"]
    train_indices = canonical_payload["development_train_indices"]
    val_indices = canonical_payload["development_val_indices"]

    assert canonical_payload["split_id"] == CANONICAL_CIFAR100_SPLIT_ID
    assert canonical_payload["generator_version"] == "canonical-cifar100-split-v1"
    assert canonical_payload["seed"] == 42
    assert len(train_indices) == 45_000
    assert len(val_indices) == 5_000
    assert train_indices == sorted(train_indices)
    assert val_indices == sorted(val_indices)
    assert set(train_indices).isdisjoint(val_indices)
    assert set(train_indices) | set(val_indices) == set(range(50_000))
    assert Counter(train_labels[index] for index in train_indices) == {
        class_id: 450 for class_id in range(100)
    }
    assert Counter(train_labels[index] for index in val_indices) == {
        class_id: 50 for class_id in range(100)
    }
    assert canonical_payload["official_test_indices"] == list(range(10_000))


def test_canonical_split_adapts_to_loader_contract_without_a_forget_set(
    canonical_payload: dict[str, object],
) -> None:
    adapted = canonical_training_payload(canonical_payload)

    assert adapted["request_type"] == "canonical_baseline"
    assert adapted["canonical_split_digest"] == canonical_payload["digest"]
    assert adapted["forget_indices"] == []
    assert adapted["retain_train_indices"] == canonical_payload[
        "development_train_indices"
    ]
    assert adapted["retain_val_indices"] == canonical_payload[
        "development_val_indices"
    ]
    assert adapted["finetune_train_indices"] == canonical_payload[
        "development_train_indices"
    ]
    assert adapted["test_all_indices"] == list(range(10_000))
    assert adapted["test_retain_indices"] == list(range(10_000))


def test_canonical_split_is_reproducible_and_unaffected_by_forget_requests() -> None:
    train_labels = _official_cifar100_labels(per_class=500)
    test_labels = _official_cifar100_labels(per_class=100)
    before_request = build_canonical_cifar100_split(train_labels, test_labels)

    make_splits(
        train_labels,
        test_labels,
        SplitConfig(
            forget_classes=[70],
            forget_fraction=1.0,
            retain_val_fraction=0.1,
            seed=42,
        ),
    )
    make_splits(
        train_labels,
        test_labels,
        SplitConfig(
            forget_classes=[92],
            forget_fraction=0.5,
            retain_val_fraction=0.1,
            seed=99,
        ),
    )

    after_requests = build_canonical_cifar100_split(train_labels, test_labels)
    assert after_requests == before_request


def test_canonical_split_rejects_tampering_and_writes_atomically(
    canonical_payload: dict[str, object], tmp_path: Path
) -> None:
    output_path = tmp_path / "canonical.json"
    write_canonical_cifar100_split(canonical_payload, output_path)
    assert load_canonical_cifar100_split(output_path) == canonical_payload
    with pytest.raises(FileExistsError):
        write_canonical_cifar100_split(canonical_payload, output_path)

    tampered = copy.deepcopy(canonical_payload)
    tampered["development_val_indices"][0] = tampered["development_val_indices"][1]
    with pytest.raises(ValueError, match="indices"):
        validate_canonical_cifar100_split(tampered)


def test_canonical_split_rejects_noncanonical_seed() -> None:
    with pytest.raises(ValueError, match="fixed at seed 42"):
        build_canonical_cifar100_split(
            _official_cifar100_labels(per_class=500),
            _official_cifar100_labels(per_class=100),
            seed=7,
        )


def test_canonical_split_binds_to_the_loaded_source_label_order(
    canonical_payload: dict[str, object],
) -> None:
    train_labels = _official_cifar100_labels(per_class=500)
    test_labels = _official_cifar100_labels(per_class=100)
    validate_canonical_cifar100_dataset_labels(
        canonical_payload, train_labels, test_labels
    )

    reordered_train_labels = list(train_labels)
    reordered_train_labels[0], reordered_train_labels[500] = (
        reordered_train_labels[500],
        reordered_train_labels[0],
    )
    with pytest.raises(ValueError, match="loaded dataset"):
        validate_canonical_cifar100_dataset_labels(
            canonical_payload, reordered_train_labels, test_labels
        )


def test_generate_canonical_split_cli_help_runs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts/generate_canonical_split.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 0
    assert "usage:" in proc.stdout.lower()
    assert "no forget request" in proc.stdout.lower()
