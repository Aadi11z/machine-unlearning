from __future__ import annotations

import pytest

from unml.pilots import (
    build_pilot_report,
    confirmation_specs,
    optimization_screen_specs,
    rank_pilot_results,
    schema_screen_specs,
)


def test_schema_screen_matches_phase_one_candidates() -> None:
    specs = schema_screen_specs()
    assert [(spec.lora_rank, spec.lora_targets) for spec in specs] == [
        (4, ("q_proj", "v_proj")),
        (8, ("q_proj", "v_proj")),
        (16, ("q_proj", "v_proj")),
        (8, ("q_proj", "k_proj", "v_proj")),
    ]
    assert all(spec.lora_alpha == spec.lora_rank for spec in specs)


def test_followup_stage_sizes_are_bounded() -> None:
    leading = schema_screen_specs()[:2]
    optimization = optimization_screen_specs(leading)
    confirmation = confirmation_specs(leading)
    assert len(optimization) == 18
    assert len(confirmation) == 6
    assert {spec.seed for spec in confirmation} == {42, 123, 456}


def test_ranking_prefers_smaller_near_tie_without_test_metrics() -> None:
    ranked = rank_pilot_results(
        [
            {"name": "large", "retain_val_acc": 0.80, "adapter_size_bytes": 200},
            {"name": "small", "retain_val_acc": 0.796, "adapter_size_bytes": 100},
        ]
    )
    assert ranked[0]["name"] == "small"
    assert ranked[0]["rank"] == 1


def test_ranking_rejects_test_metrics() -> None:
    with pytest.raises(ValueError, match="test-set metrics"):
        rank_pilot_results([{"retain_val_acc": 0.8, "test_all_acc": 0.7}])


def test_report_declares_validation_only() -> None:
    report = build_pilot_report(
        split_path="split.json",
        split_digest="abc",
        prompt_contract={"digest": "prompt"},
        results=[{"name": "r8", "retain_val_acc": 0.8}],
    )
    assert report["schema"] == "unml-lora-pilot-report-v1"
    assert report["test_metrics_used"] is False
    assert report["winner"]["name"] == "r8"
