from __future__ import annotations

from pathlib import Path

import pytest

import pandas as pd

from unml.probe_ui import (
    _model_load_error_message,
    _result_html,
    load_probe_ui_settings,
)


def test_probe_ui_settings_resolve_only_relative_checkpoint_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "parameters.yaml"
    config_path.write_text(
        """
outputs:
  root: outputs/{dataset}
probe_ui:
  dataset: cifar100
  request: rose_selective
  target_class: rose
  comparison: comparison/comparison.csv
  baseline:
    name: Fine-tuned baseline
    checkpoint: baseline.pt
    comparison_model: finetuned
  candidates:
    - name: H-TGSD
      checkpoint: h_tgsd.pt
      comparison_model: h_tgsd
""",
        encoding="utf-8",
    )

    settings = load_probe_ui_settings(
        str(config_path),
        output_root_override=str(tmp_path / "artifacts"),
    )

    assert settings.output_root == tmp_path / "artifacts"
    assert settings.baseline.checkpoint_path == tmp_path / "artifacts" / "baseline.pt"
    assert settings.candidates[0].checkpoint_path == tmp_path / "artifacts" / "h_tgsd.pt"


def test_probe_ui_settings_reject_parent_checkpoint_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "parameters.yaml"
    config_path.write_text(
        """
probe_ui:
  dataset: cifar100
  request: rose_selective
  target_class: rose
  comparison: comparison.csv
  baseline:
    name: Fine-tuned baseline
    checkpoint: ../baseline.pt
    comparison_model: finetuned
  candidates:
    - name: H-TGSD
      checkpoint: h_tgsd.pt
      comparison_model: h_tgsd
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relative path"):
        load_probe_ui_settings(str(config_path))


def test_probe_ui_settings_reject_missing_comparison_path(tmp_path: Path) -> None:
    config_path = tmp_path / "parameters.yaml"
    config_path.write_text(
        """
probe_ui:
  dataset: cifar100
  request: rose_selective
  target_class: rose
  baseline:
    name: Fine-tuned baseline
    checkpoint: baseline.pt
    comparison_model: finetuned
  candidates:
    - name: H-TGSD
      checkpoint: h_tgsd.pt
      comparison_model: h_tgsd
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-empty relative path"):
        load_probe_ui_settings(str(config_path))


def test_offline_model_load_error_explains_how_to_cache_clip() -> None:
    message = _model_load_error_message(OSError("cache miss"), offline=True)

    assert "cache_model.py --dataset cifar100" in message
    assert "--offline" in message


def test_result_html_separates_probe_from_saved_evaluation() -> None:
    result = _result_html(
        target_class_name="rose",
        candidate_name="H-TGSD",
        baseline_prediction={
            "target_probability": 0.9997,
            "target_rank": 1,
            "top_k": [{"class_name": "rose", "probability": 0.9997}],
        },
        candidate_prediction={
            "target_probability": 0.0086,
            "target_rank": 77,
            "top_k": [{"class_name": "cloud", "probability": 0.0151}],
        },
        metrics=pd.Series(
            {
                "target_test_acc": 0.11,
                "sibling_test_acc": 0.9175,
                "utility_test_all": 0.8557,
                "mia_resistance_confidence": 0.7038,
                "mia_auc_confidence": 0.3519,
            }
        ),
    )

    assert "Target suppression observed" in result
    assert "rank 1 to 77" in result
    assert "11.0%" in result
    assert "70.4%" in result
    assert "illustrative external probe" in result
