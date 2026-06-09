from __future__ import annotations

import json

import pytest

from unml.config import (
    load_runtime_config,
    normalize_dataset_name,
    resolve_data_dir,
    resolve_dataset_and_split_path,
    resolve_dataset_value,
    resolve_forget_classes,
    resolve_output_root,
    resolve_stage_output_dir,
)
from unml.data import (
    CIFAR100_CLASSES,
    download_and_prepare_splits,
    get_dataset_spec,
    load_split_metadata,
    validate_checkpoint_dataset,
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
    assert split_path == "outputs/cifar100/splits/cifar100_split.json"
    assert resolve_forget_classes(None, payload, dataset_name) == "54,62,70,82,92"
    assert resolve_output_root(None, payload, dataset_name).as_posix() == (
        "outputs/cifar100"
    )
    assert resolve_stage_output_dir(
        None, payload, dataset_name, "training"
    ).as_posix() == "outputs/cifar100/finetune"


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
        "/scratch/user/project/outputs/cifar100/splits/cifar100_split.json"
    )
    assert resolve_stage_output_dir(
        None, payload, dataset_name, "unlearning"
    ).as_posix() == "/scratch/user/project/outputs/cifar100/unlearning"


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
