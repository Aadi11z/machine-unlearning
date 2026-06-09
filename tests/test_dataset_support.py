from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import unml.data as data_module
from unml.config import (
    load_runtime_config,
    normalize_dataset_name,
    resolve_data_dir,
    resolve_dataset_and_split_path,
    resolve_dataset_value,
    resolve_forget_classes,
    resolve_model_value,
    resolve_output_root,
    resolve_section_str_list,
    resolve_section_value,
    resolve_stage_output_dir,
)
from unml.data import (
    CIFAR100_CLASSES,
    CIFAR100_SUPERCLASSES,
    SplitConfig,
    download_and_prepare_splits,
    get_dataset_spec,
    load_split_metadata,
    make_splits,
    validate_checkpoint_dataset,
    validate_hierarchy_request,
)


def _write_split(tmp_path, payload: dict) -> str:
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    return str(split_path)


def test_dataset_registry_contains_complete_cifar_vocabularies() -> None:
    assert len(get_dataset_spec("cifar-10").class_names) == 10
    assert len(get_dataset_spec("CIFAR_100").class_names) == 100
    assert [CIFAR100_CLASSES[index] for index in (54, 62, 70, 82, 92)] == [
        "orchid",
        "poppy",
        "rose",
        "sunflower",
        "tulip",
    ]


def test_cifar100_superclasses_cover_every_class_once() -> None:
    superclass_members = [
        class_id
        for class_ids in CIFAR100_SUPERCLASSES.values()
        for class_id in class_ids
    ]

    assert len(CIFAR100_SUPERCLASSES) == 20
    assert len(superclass_members) == 100
    assert sorted(superclass_members) == list(range(100))
    assert CIFAR100_SUPERCLASSES["flowers"] == (54, 62, 70, 82, 92)


def test_dataset_override_gets_matching_split_and_forget_classes() -> None:
    payload = {
        "data": {
            "dataset": "cifar10",
            "profiles": {
                "cifar10": {
                    "forget_classes": "3,5",
                    "forget_fraction": 1.0,
                },
                "cifar100": {
                    "forget_classes": "54,62,70,82,92",
                    "forget_fraction": 0.5,
                },
            },
        },
        "outputs": {"root": "outputs/{dataset}"},
    }

    dataset_name, split_path = resolve_dataset_and_split_path(
        "cifar100", None, payload
    )

    assert dataset_name == "cifar100"
    assert split_path == "outputs/cifar100/splits/cifar100_split.json"
    assert resolve_forget_classes(None, payload, dataset_name) == "54,62,70,82,92"
    assert resolve_forget_classes("1,2", payload, dataset_name) == "1,2"
    assert (
        resolve_dataset_value(
            None, payload, dataset_name, "forget_fraction", 1.0
        )
        == 0.5
    )


def test_parameters_yaml_switches_complete_dataset_profile(monkeypatch) -> None:
    monkeypatch.delenv("UNML_DATA", raising=False)
    monkeypatch.delenv("UNML_OUTPUTS", raising=False)
    payload = load_runtime_config("config/parameters.yaml")

    payload["data"]["dataset"] = "cifar100"
    dataset_name, split_path = resolve_dataset_and_split_path(None, None, payload)

    assert dataset_name == "cifar100"
    assert split_path == (
        "outputs/cifar100/flowers_superclass/splits/"
        "flowers_superclass_split.json"
    )
    assert resolve_forget_classes(None, payload, dataset_name) == "54,62,70,82,92"
    assert resolve_output_root(None, payload, dataset_name).as_posix() == (
        "outputs/cifar100/flowers_superclass"
    )
    assert resolve_stage_output_dir(
        None, payload, dataset_name, "training"
    ).as_posix() == "outputs/cifar100/flowers_superclass/finetune"

    rose_dataset, rose_split = resolve_dataset_and_split_path(
        None, None, payload, "rose_selective"
    )
    assert rose_dataset == "cifar100"
    assert rose_split == (
        "outputs/cifar100/rose_selective/splits/rose_selective_split.json"
    )
    assert (
        resolve_forget_classes(
            None, payload, rose_dataset, "rose_selective"
        )
        == "70"
    )
    assert (
        resolve_model_value(
            None, payload, dataset_name, "model_name", "fallback"
        )
        == "openai/clip-vit-base-patch16"
    )
    assert (
        resolve_model_value(
            None, payload, dataset_name, "adapter_type", "fallback"
        )
        == "vision_lora"
    )
    assert (
        resolve_section_value(
            None,
            payload,
            dataset_name,
            "training",
            "batch_size",
            128,
        )
        == 64
    )
    assert (
        resolve_section_value(
            None,
            payload,
            dataset_name,
            "training",
            "gradient_accumulation_steps",
            1,
        )
        == 2
    )
    cifar100_methods = resolve_section_str_list(
        None,
        payload,
        dataset_name,
        "unlearning",
        "methods",
        (),
    )
    assert "h_tgsd" in cifar100_methods
    assert "h_tgsd_no_sibling_preservation" in cifar100_methods
    assert "h_tgsd" not in resolve_section_str_list(
        None,
        payload,
        "cifar10",
        "unlearning",
        "methods",
        (),
    )


def test_sharanga_environment_redirects_heavy_artifacts(monkeypatch) -> None:
    payload = load_runtime_config("config/parameters.yaml")
    monkeypatch.setenv("UNML_DATA", "/scratch/user/project/data")
    monkeypatch.setenv("UNML_OUTPUTS", "/scratch/user/project/outputs")

    dataset_name, split_path = resolve_dataset_and_split_path(
        "cifar100", None, payload
    )

    assert resolve_data_dir(None, payload).as_posix() == (
        "/scratch/user/project/data"
    )
    assert split_path == (
        "/scratch/user/project/outputs/cifar100/flowers_superclass/"
        "splits/flowers_superclass_split.json"
    )
    assert resolve_stage_output_dir(
        None, payload, dataset_name, "unlearning"
    ).as_posix() == (
        "/scratch/user/project/outputs/cifar100/flowers_superclass/unlearning"
    )


def test_flowers_superclass_request_builds_grouped_partitions() -> None:
    spec = get_dataset_spec("cifar100")
    train_labels = list(range(100)) * 2
    test_labels = list(range(100))
    cfg = SplitConfig(
        forget_classes=[54, 62, 70, 82, 92],
        forget_fraction=1.0,
        retain_val_fraction=0.1,
        seed=42,
        request_name="flowers_superclass",
        request_type="superclass",
        superclass="flowers",
    )
    siblings, unrelated = validate_hierarchy_request(spec, cfg)
    cfg.sibling_classes = siblings
    cfg.unrelated_classes = unrelated

    split = make_splits(train_labels, test_labels, cfg)

    assert siblings == []
    assert len(unrelated) == 95
    assert {
        train_labels[index] for index in split["forget_indices"]
    } == {54, 62, 70, 82, 92}
    assert split["test_sibling_indices"] == []
    assert {
        test_labels[index] for index in split["test_unrelated_indices"]
    } == set(unrelated)


def test_rose_selective_request_preserves_flower_siblings() -> None:
    spec = get_dataset_spec("cifar100")
    train_labels = list(range(100)) * 2
    test_labels = list(range(100))
    cfg = SplitConfig(
        forget_classes=[70],
        forget_fraction=1.0,
        retain_val_fraction=0.1,
        seed=42,
        request_name="rose_selective",
        request_type="selective_class",
        superclass="flowers",
        sibling_classes=[54, 62, 82, 92],
    )
    siblings, unrelated = validate_hierarchy_request(spec, cfg)
    cfg.sibling_classes = siblings
    cfg.unrelated_classes = unrelated

    split = make_splits(train_labels, test_labels, cfg)

    assert split["target_classes"] == [70]
    assert split["sibling_classes"] == [54, 62, 82, 92]
    assert len(split["unrelated_classes"]) == 95
    assert {
        test_labels[index] for index in split["test_forget_indices"]
    } == {70}
    assert {
        test_labels[index] for index in split["test_sibling_indices"]
    } == {54, 62, 82, 92}
    assert set(split["test_forget_indices"]).isdisjoint(
        split["test_sibling_indices"]
    )
    assert set(split["test_sibling_indices"]).isdisjoint(
        split["test_unrelated_indices"]
    )


def test_request_split_artifact_records_hierarchy_metadata(
    tmp_path, monkeypatch
) -> None:
    train_labels = list(range(100)) * 2
    test_labels = list(range(100))
    monkeypatch.setattr(
        data_module,
        "_load_dataset_pair",
        lambda *_args, **_kwargs: (
            SimpleNamespace(targets=train_labels),
            SimpleNamespace(targets=test_labels),
        ),
    )
    split_path = tmp_path / "rose_selective_split.json"

    result = download_and_prepare_splits(
        data_dir=str(tmp_path / "data"),
        split_path=str(split_path),
        forget_classes=[70],
        forget_fraction=1.0,
        retain_val_fraction=0.1,
        seed=42,
        dataset_name="cifar100",
        request_name="rose_selective",
        request_type="selective_class",
        superclass="flowers",
        sibling_classes=[54, 62, 82, 92],
    )
    stored = json.loads(split_path.read_text(encoding="utf-8"))

    assert stored == result
    assert stored["request_name"] == "rose_selective"
    assert stored["request_type"] == "selective_class"
    assert stored["superclass"] == "flowers"
    assert stored["target_classes"] == [70]
    assert stored["sibling_classes"] == [54, 62, 82, 92]
    assert len(stored["unrelated_classes"]) == 95
    assert stored["seed"] == 42
    assert stored["forget_fraction"] == 1.0


def test_legacy_split_defaults_to_cifar10(tmp_path) -> None:
    split_path = _write_split(tmp_path, {"forget_classes": [3, 5]})

    _, spec, class_names = load_split_metadata(split_path, "cifar10")

    assert spec.name == "cifar10"
    assert len(class_names) == 10


def test_split_and_checkpoint_dataset_mismatches_are_rejected(tmp_path) -> None:
    split_path = _write_split(
        tmp_path,
        {
            "dataset": "cifar100",
            "class_names": list(CIFAR100_CLASSES),
        },
    )

    with pytest.raises(ValueError, match="does not match"):
        load_split_metadata(split_path, "cifar10")

    with pytest.raises(ValueError, match="created for cifar10"):
        validate_checkpoint_dataset(
            {"dataset": "cifar10"}, "cifar100", "checkpoint.pt"
        )


def test_invalid_cifar100_forget_class_is_rejected_before_download(tmp_path) -> None:
    with pytest.raises(ValueError, match="expected ids from 0 to 99"):
        download_and_prepare_splits(
            data_dir=str(tmp_path / "data"),
            split_path=str(tmp_path / "split.json"),
            forget_classes=[100],
            forget_fraction=1.0,
            retain_val_fraction=0.1,
            seed=42,
            dataset_name="cifar100",
        )


def test_unsupported_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported dataset"):
        normalize_dataset_name("imagenet")
