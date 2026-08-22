from __future__ import annotations

import json
from pathlib import Path

import pytest

from unml.data import get_dataset_spec, make_splits
from unml.request_factory import (
    SelectiveRequest,
    list_superclass_groups,
    resolve_selective_request,
    selective_split_config,
    validate_selective_request,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ROSE_SPLIT_PATH = (
    REPO_ROOT
    / "outputs"
    / "cifar100"
    / "rose_selective"
    / "splits"
    / "rose_selective_split.json"
)


def test_resolve_rose_matches_recorded_request() -> None:
    request = resolve_selective_request("cifar100", 70)
    assert request.class_id == 70
    assert request.class_name == "rose"
    assert request.superclass == "flowers"
    assert request.sibling_classes == (54, 62, 82, 92)
    assert request.request_name == "rose_selective"


def test_resolve_accepts_names_and_underscore_variants() -> None:
    by_name = resolve_selective_request("cifar100", "rose")
    by_variant = resolve_selective_request("cifar100", "Aquarium_Fish")
    assert by_name.class_id == 70
    assert by_variant.class_id == 1
    assert by_variant.request_name == "aquarium_fish_selective"


def test_every_cifar100_class_resolves_with_valid_siblings() -> None:
    spec = get_dataset_spec("cifar100")
    seen_names = set()
    for class_id in range(len(spec.class_names)):
        request = resolve_selective_request("cifar100", class_id)
        assert request.superclass in spec.superclasses
        assert len(request.sibling_classes) == 4
        assert request.class_id not in request.sibling_classes
        superclass_members = set(spec.superclasses[request.superclass])
        assert set(request.sibling_classes) < superclass_members
        assert request.request_name not in seen_names
        seen_names.add(request.request_name)
    assert len(seen_names) == len(spec.class_names)


@pytest.mark.parametrize(
    ("dataset_name", "target"),
    [
        ("cifar100", -1),
        ("cifar100", 100),
        ("cifar100", "not_a_class"),
        ("cifar100", ""),
        ("cifar10", 3),
    ],
)
def test_invalid_targets_raise(dataset_name: str, target) -> None:
    with pytest.raises(ValueError):
        resolve_selective_request(dataset_name, target)


def test_factory_split_config_passes_hierarchy_validation() -> None:
    spec = get_dataset_spec("cifar100")
    request = resolve_selective_request("cifar100", 70)
    cfg = selective_split_config(
        request,
        forget_fraction=1.0,
        retain_val_fraction=0.1,
        seed=42,
    )
    validate_selective_request(spec, cfg)
    assert cfg.unrelated_classes == ()


def test_factory_split_groups_match_make_splits() -> None:
    spec = get_dataset_spec("cifar100")
    request = resolve_selective_request("cifar100", 70)

    def labels_for(class_ids: list[int], per_class: int, offset: int = 0) -> list[int]:
        labels: list[int] = []
        for local_id, class_id in enumerate(class_ids):
            labels.extend([class_id] * per_class)
            offset += per_class
        return labels

    train_classes = [70, 54, 62, 82, 92, 0, 1, 33]
    train_labels = labels_for(train_classes, per_class=4)
    test_labels = labels_for([70, 54, 92, 0, 33], per_class=2)

    cfg = selective_split_config(
        request,
        forget_fraction=1.0,
        retain_val_fraction=0.25,
        seed=42,
    )
    splits = make_splits(
        train_labels=train_labels, test_labels=test_labels, cfg=cfg
    )

    assert all(
        train_labels[index] == 70 for index in splits["forget_indices"]
    )
    assert set(splits["sibling_classes"]) == {54, 62, 82, 92}
    sibling_train = splits["sibling_retain_train_indices"]
    assert sibling_train
    assert all(
        train_labels[index] in {54, 62, 82, 92} for index in sibling_train
    )
    unrelated_train = splits["unrelated_retain_train_indices"]
    assert all(
        train_labels[index] not in {70, 54, 62, 82, 92}
        for index in unrelated_train
    )
    assert all(
        test_labels[index] == 70 for index in splits["test_forget_indices"]
    )
    assert all(
        test_labels[index] in {54, 62, 82, 92}
        for index in splits["test_sibling_indices"]
    )


def test_superclass_group_listing_is_complete() -> None:
    groups = list_superclass_groups("cifar100")
    spec = get_dataset_spec("cifar100")
    listed = sorted(class_id for group in groups for class_id in group["class_ids"])
    assert listed == list(range(len(spec.class_names)))
    flowers = next(g for g in groups if g["superclass"] == "flowers")
    assert flowers["class_names"] == ["orchid", "poppy", "rose", "sunflower", "tulip"]


def test_rose_split_artifact_matches_factory_fields() -> None:
    if not ROSE_SPLIT_PATH.is_file():
        pytest.skip("recorded rose split artifact is not present")
    stored = json.loads(ROSE_SPLIT_PATH.read_text())
    request = resolve_selective_request("cifar100", 70)
    assert stored["request_name"] == request.request_name
    assert stored["request_type"] == "selective_class"
    assert stored["superclass"] == request.superclass
    assert stored["target_classes"] == [request.class_id]
    assert stored["sibling_classes"] == list(request.sibling_classes)


def test_job_config_builder_resolves_cifar100_profile() -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from unlearn_job import build_job_unlearn_config
    finally:
        sys.path.pop(0)

    from unml.config import load_runtime_config
    from unml.unlearn import UnlearnConfig

    runtime_cfg = load_runtime_config(
        str(REPO_ROOT / "config" / "parameters.yaml")
    )
    request = resolve_selective_request("cifar100", 70)
    cfg = build_job_unlearn_config(
        runtime_cfg,
        request=request,
        dataset_name="cifar100",
        data_dir="data",
        split_path="splits.json",
        finetuned_checkpoint="baseline.pt",
        output_dir="job_out",
        method="ga_kl",
        steps=123,
        seed=7,
        device="cpu",
    )
    assert isinstance(cfg, UnlearnConfig)
    assert cfg.model_name == "openai/clip-vit-base-patch16"
    assert cfg.train_logit_scale is False
    assert cfg.steps == 123
    assert cfg.seed == 7
    assert cfg.method == "ga_kl"
    assert cfg.split_path == "splits.json"
