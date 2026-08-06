from __future__ import annotations

from pathlib import Path

import pytest

from unml.probe_ui import load_probe_ui_settings


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
