from __future__ import annotations

import sys

import pytest

import scripts.generate_canonical_oracle_split as oracle_split_script
from scripts.generate_canonical_oracle_split import resolve_oracle_request


def test_flowers_oracle_request_resolves_complete_hierarchy() -> None:
    request = resolve_oracle_request(
        config_path="config/parameters.yaml",
        request_name="flowers_superclass",
    )

    assert request == {
        "request_name": "flowers_superclass",
        "request_type": "superclass",
        "superclass": "flowers",
        "target_classes": [54, 62, 70, 82, 92],
        "sibling_classes": [],
    }


def test_selective_oracle_request_resolves_siblings() -> None:
    request = resolve_oracle_request(
        config_path="config/parameters.yaml",
        request_name="rose_selective",
    )

    assert request["request_type"] == "selective_class"
    assert request["target_classes"] == [70]
    assert request["sibling_classes"] == [54, 62, 82, 92]


def test_oracle_request_rejects_class_override_drift() -> None:
    with pytest.raises(ValueError, match="does not match configured"):
        resolve_oracle_request(
            config_path="config/parameters.yaml",
            request_name="flowers_superclass",
            forget_classes_override="70",
        )


def test_oracle_split_entrypoint_passes_hierarchy_metadata(
    tmp_path, monkeypatch
) -> None:
    captured = {}
    monkeypatch.setattr(
        oracle_split_script,
        "download_and_prepare_splits",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        oracle_split_script,
        "summarize_splits",
        lambda _path: {"forget_train": 2500},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_canonical_oracle_split.py",
            "--data-dir",
            str(tmp_path / "data"),
            "--output-path",
            str(tmp_path / "split.json"),
            "--request",
            "flowers_superclass",
        ],
    )

    oracle_split_script.main()

    assert captured["request_name"] == "flowers_superclass"
    assert captured["request_type"] == "superclass"
    assert captured["superclass"] == "flowers"
    assert captured["forget_classes"] == [54, 62, 70, 82, 92]
    assert captured["sibling_classes"] == []
