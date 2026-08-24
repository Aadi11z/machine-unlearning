"""Target-neutral, reproducible CIFAR-100 development split generation.

This module deliberately has no concept of a forget request.  It owns the
single seed-42, class-stratified 45k/5k development partition required before
baseline pilots.  The official CIFAR-100 test set is recorded only as the
untouched evaluation set.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import CIFAR100_CLASSES, get_dataset_spec


CANONICAL_CIFAR100_SPLIT_ID = "cifar100_canonical_development_v1"
CANONICAL_CIFAR100_SPLIT_VERSION = 1
CANONICAL_CIFAR100_GENERATOR_VERSION = "canonical-cifar100-split-v1"
CANONICAL_CIFAR100_SEED = 42
CANONICAL_CIFAR100_TRAIN_PER_CLASS = 450
CANONICAL_CIFAR100_VAL_PER_CLASS = 50
CANONICAL_CIFAR100_SOURCE_TRAIN_PER_CLASS = (
    CANONICAL_CIFAR100_TRAIN_PER_CLASS + CANONICAL_CIFAR100_VAL_PER_CLASS
)
CANONICAL_CIFAR100_SOURCE_TEST_PER_CLASS = 100


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return the stable encoding used for split digests."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_split_digest(payload: Mapping[str, Any]) -> str:
    """Hash split metadata, excluding its self-referential ``digest`` field."""
    digest_payload = dict(payload)
    digest_payload.pop("digest", None)
    return hashlib.sha256(_canonical_json_bytes(digest_payload)).hexdigest()


def class_vocabulary_digest(class_names: Sequence[str] = CIFAR100_CLASSES) -> str:
    """Return a stable digest for the ordered CIFAR-100 vocabulary."""
    return hashlib.sha256(
        _canonical_json_bytes({"class_names": [str(name) for name in class_names]})
    ).hexdigest()


def _normalise_labels(labels: Sequence[int], *, split_name: str) -> list[int]:
    normalised = [int(label) for label in labels]
    invalid = sorted({label for label in normalised if label < 0 or label >= 100})
    if invalid:
        raise ValueError(
            f"{split_name} labels must be CIFAR-100 class ids in [0, 99]; "
            f"found {invalid}"
        )
    return normalised


def _require_expected_distribution(
    labels: Sequence[int],
    *,
    split_name: str,
    expected_total: int,
    expected_per_class: int,
) -> None:
    if len(labels) != expected_total:
        raise ValueError(
            f"{split_name} must contain {expected_total} examples; found {len(labels)}"
        )
    counts = Counter(labels)
    incorrect = {
        class_id: counts.get(class_id, 0)
        for class_id in range(100)
        if counts.get(class_id, 0) != expected_per_class
    }
    if incorrect:
        raise ValueError(
            f"{split_name} must contain {expected_per_class} examples per class; "
            f"mismatches: {incorrect}"
        )


def build_canonical_cifar100_split(
    train_labels: Sequence[int],
    test_labels: Sequence[int],
    *,
    seed: int = CANONICAL_CIFAR100_SEED,
) -> dict[str, Any]:
    """Build the fixed, target-neutral CIFAR-100 45k/5k development split.

    The split is stratified with exactly 450 training and 50 validation samples
    per class.  The returned index lists are sorted in source-dataset order;
    the seeded sampling only chooses the validation members.  No forget-set
    argument exists, intentionally, so later unlearning requests cannot alter
    this artifact.
    """
    if seed != CANONICAL_CIFAR100_SEED:
        raise ValueError(
            "The canonical CIFAR-100 split is fixed at seed "
            f"{CANONICAL_CIFAR100_SEED}; received {seed}"
        )

    normalised_train_labels = _normalise_labels(train_labels, split_name="train")
    normalised_test_labels = _normalise_labels(test_labels, split_name="test")
    _require_expected_distribution(
        normalised_train_labels,
        split_name="train",
        expected_total=50_000,
        expected_per_class=CANONICAL_CIFAR100_SOURCE_TRAIN_PER_CLASS,
    )
    _require_expected_distribution(
        normalised_test_labels,
        split_name="test",
        expected_total=10_000,
        expected_per_class=CANONICAL_CIFAR100_SOURCE_TEST_PER_CLASS,
    )

    validation_indices: list[int] = []
    for class_id in range(100):
        class_indices = [
            index
            for index, label in enumerate(normalised_train_labels)
            if label == class_id
        ]
        # A distinct, arithmetic seed avoids coupling a class's samples to
        # ordering or implementation changes in another class's random draw.
        class_rng = random.Random(seed + class_id * 1_000_003)
        validation_indices.extend(
            class_rng.sample(class_indices, CANONICAL_CIFAR100_VAL_PER_CLASS)
        )

    development_val_indices = sorted(validation_indices)
    validation_set = set(development_val_indices)
    development_train_indices = [
        index
        for index in range(len(normalised_train_labels))
        if index not in validation_set
    ]

    class_counts = [
        {
            "class_id": class_id,
            "class_name": CIFAR100_CLASSES[class_id],
            "source_train_count": CANONICAL_CIFAR100_SOURCE_TRAIN_PER_CLASS,
            "development_train_count": CANONICAL_CIFAR100_TRAIN_PER_CLASS,
            "development_val_count": CANONICAL_CIFAR100_VAL_PER_CLASS,
            "official_test_count": CANONICAL_CIFAR100_SOURCE_TEST_PER_CLASS,
        }
        for class_id in range(100)
    ]
    payload: dict[str, Any] = {
        "schema_version": CANONICAL_CIFAR100_SPLIT_VERSION,
        "generator_version": CANONICAL_CIFAR100_GENERATOR_VERSION,
        "split_id": CANONICAL_CIFAR100_SPLIT_ID,
        "dataset": "cifar100",
        "dataset_source": "torchvision.datasets.CIFAR100",
        "seed": seed,
        "class_names": list(CIFAR100_CLASSES),
        "class_vocabulary_digest": class_vocabulary_digest(),
        "train_labels": normalised_train_labels,
        "test_labels": normalised_test_labels,
        "development_train_indices": development_train_indices,
        "development_val_indices": development_val_indices,
        "official_test_indices": list(range(len(normalised_test_labels))),
        "stratification": {
            "source_train_per_class": CANONICAL_CIFAR100_SOURCE_TRAIN_PER_CLASS,
            "development_train_per_class": CANONICAL_CIFAR100_TRAIN_PER_CLASS,
            "development_val_per_class": CANONICAL_CIFAR100_VAL_PER_CLASS,
            "official_test_per_class": CANONICAL_CIFAR100_SOURCE_TEST_PER_CLASS,
            "class_counts": class_counts,
        },
    }
    payload["digest"] = canonical_split_digest(payload)
    return payload


def validate_canonical_cifar100_split(payload: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless payload satisfies the canonical split contract."""
    if payload.get("schema_version") != CANONICAL_CIFAR100_SPLIT_VERSION:
        raise ValueError("Unsupported canonical CIFAR-100 split schema version")
    if payload.get("generator_version") != CANONICAL_CIFAR100_GENERATOR_VERSION:
        raise ValueError("Unsupported canonical CIFAR-100 split generator version")
    if payload.get("split_id") != CANONICAL_CIFAR100_SPLIT_ID:
        raise ValueError("Unexpected canonical CIFAR-100 split id")
    if payload.get("dataset") != "cifar100":
        raise ValueError("Canonical split dataset must be cifar100")
    if payload.get("seed") != CANONICAL_CIFAR100_SEED:
        raise ValueError("Canonical CIFAR-100 split seed must be 42")

    class_names = payload.get("class_names")
    if class_names != list(CIFAR100_CLASSES):
        raise ValueError("Canonical CIFAR-100 vocabulary does not match the contract")
    if payload.get("class_vocabulary_digest") != class_vocabulary_digest():
        raise ValueError("Canonical CIFAR-100 vocabulary digest does not match")

    train_labels = _normalise_labels(payload.get("train_labels", []), split_name="train")
    test_labels = _normalise_labels(payload.get("test_labels", []), split_name="test")
    _require_expected_distribution(
        train_labels,
        split_name="train",
        expected_total=50_000,
        expected_per_class=CANONICAL_CIFAR100_SOURCE_TRAIN_PER_CLASS,
    )
    _require_expected_distribution(
        test_labels,
        split_name="test",
        expected_total=10_000,
        expected_per_class=CANONICAL_CIFAR100_SOURCE_TEST_PER_CLASS,
    )

    train_indices = payload.get("development_train_indices", [])
    val_indices = payload.get("development_val_indices", [])
    test_indices = payload.get("official_test_indices", [])
    if not all(isinstance(index, int) for index in train_indices + val_indices + test_indices):
        raise ValueError("Canonical split indices must be integers")
    if train_indices != sorted(train_indices) or val_indices != sorted(val_indices):
        raise ValueError("Canonical development indices must be sorted")
    if len(train_indices) != 45_000 or len(val_indices) != 5_000:
        raise ValueError("Canonical development split must contain 45k train and 5k val")
    if set(train_indices) & set(val_indices):
        raise ValueError("Canonical development train and validation indices overlap")
    if set(train_indices) | set(val_indices) != set(range(50_000)):
        raise ValueError("Canonical development indices must partition official training data")
    if test_indices != list(range(10_000)):
        raise ValueError("Canonical split must preserve the untouched official test indices")

    for class_id in range(100):
        if sum(train_labels[index] == class_id for index in train_indices) != 450:
            raise ValueError(f"Canonical development train class {class_id} is unbalanced")
        if sum(train_labels[index] == class_id for index in val_indices) != 50:
            raise ValueError(f"Canonical development validation class {class_id} is unbalanced")

    expected_digest = canonical_split_digest(payload)
    if payload.get("digest") != expected_digest:
        raise ValueError("Canonical split digest does not match payload")


def validate_canonical_cifar100_dataset_labels(
    payload: Mapping[str, Any],
    train_labels: Sequence[int],
    test_labels: Sequence[int],
) -> None:
    """Verify that a split artifact belongs to the loaded CIFAR-100 dataset.

    The split digest protects the artifact itself.  Comparing the recorded
    labels with the loaded torchvision labels additionally prevents a valid
    artifact from being used with a different source-dataset ordering.
    """
    validate_canonical_cifar100_split(payload)
    if _normalise_labels(train_labels, split_name="loaded train") != payload["train_labels"]:
        raise ValueError("Canonical split train labels do not match the loaded dataset")
    if _normalise_labels(test_labels, split_name="loaded test") != payload["test_labels"]:
        raise ValueError("Canonical split test labels do not match the loaded dataset")


def canonical_training_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt a validated canonical split to the existing loader contract.

    The legacy loader expects retain/forget/test keys. A canonical baseline has
    no forget set: development training and validation are the retain partitions,
    and the complete official test set is the evaluation partition. Keeping this
    adapter explicit prevents request-specific split construction from being
    mistaken for canonical baseline data.
    """
    validate_canonical_cifar100_split(payload)
    official_test_indices = list(payload["official_test_indices"])
    development_train_indices = list(payload["development_train_indices"])
    development_val_indices = list(payload["development_val_indices"])
    return {
        "dataset": payload["dataset"],
        "class_names": list(payload["class_names"]),
        "request_name": payload["split_id"],
        "request_type": "canonical_baseline",
        "canonical_split_id": payload["split_id"],
        "canonical_split_digest": payload["digest"],
        "forget_classes": [],
        "target_classes": [],
        "sibling_classes": [],
        "unrelated_classes": list(range(len(payload["class_names"]))),
        "forget_indices": [],
        "retain_train_indices": development_train_indices,
        "retain_val_indices": development_val_indices,
        "finetune_train_indices": development_train_indices,
        "test_forget_indices": [],
        "test_retain_indices": official_test_indices,
        "test_all_indices": official_test_indices,
        "test_sibling_indices": [],
        "test_unrelated_indices": official_test_indices,
        "sibling_retain_train_indices": [],
        "unrelated_retain_train_indices": development_train_indices,
        "seed": payload["seed"],
    }


def load_canonical_cifar100_split(path: str | Path) -> dict[str, Any]:
    """Load and validate a canonical split artifact before a consumer uses it."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_canonical_cifar100_split(payload)
    return payload


def write_canonical_cifar100_split(
    payload: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically persist a validated split, refusing accidental replacement."""
    validate_canonical_cifar100_split(payload)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Canonical split already exists at {output_path}; use --overwrite only "
            "when intentionally replacing the identical, reproducible artifact"
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output_path


def generate_canonical_cifar100_split(
    data_dir: str | Path,
    output_path: str | Path,
    *,
    download: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Load official CIFAR-100 labels and write the canonical split artifact."""
    spec = get_dataset_spec("cifar100")
    train_dataset = spec.dataset_class(root=str(data_dir), train=True, download=download)
    test_dataset = spec.dataset_class(root=str(data_dir), train=False, download=download)
    payload = build_canonical_cifar100_split(train_dataset.targets, test_dataset.targets)
    write_canonical_cifar100_split(payload, output_path, overwrite=overwrite)
    return payload
