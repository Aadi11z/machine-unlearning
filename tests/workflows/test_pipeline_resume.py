from __future__ import annotations

import json
from pathlib import Path

from scripts.run_study import build_pipeline_command
from unml.pipeline import (
    run_pipeline_stage,
    stage_contract,
    stage_receipt_path,
    validate_stage_receipt,
)


def test_pipeline_stage_skips_only_identical_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "config.yaml"
    output_path = tmp_path / "checkpoint.pt"
    input_path.write_text("seed: 42\n", encoding="utf-8")
    calls = []

    def runner(command, *, check, env) -> None:
        assert check is True
        calls.append((command, env))
        output_path.write_text(f"run={len(calls)}\n", encoding="utf-8")

    arguments = {
        "stage": "finetune",
        "command": ["python", "train.py", "--seed", "42"],
        "env": {"TEST": "1"},
        "output_root": tmp_path,
        "input_paths": [input_path],
        "output_paths": [output_path],
        "resume": True,
        "runner": runner,
    }
    assert run_pipeline_stage(**arguments) == "completed"
    assert run_pipeline_stage(**arguments) == "skipped"
    assert len(calls) == 1

    output_path.write_text("manually changed\n", encoding="utf-8")
    assert run_pipeline_stage(**arguments) == "completed"
    assert len(calls) == 2

    input_path.write_text("seed: 123\n", encoding="utf-8")
    assert run_pipeline_stage(**arguments) == "completed"
    assert len(calls) == 3

    assert run_pipeline_stage(**arguments, force=True) == "completed"
    assert len(calls) == 4


def test_stage_receipt_is_machine_readable_and_detects_contract_change(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    output = tmp_path / "comparison.csv"
    config.write_text("dataset: cifar100\n", encoding="utf-8")
    output.write_text("model\nh_tgsd\n", encoding="utf-8")
    command = ["python", "evaluate.py"]

    def runner(command, *, check, env) -> None:
        return None

    run_pipeline_stage(
        stage="evaluate",
        command=command,
        env={},
        output_root=tmp_path,
        input_paths=[config],
        output_paths=[output],
        resume=False,
        runner=runner,
    )
    receipt = stage_receipt_path(tmp_path, "evaluate")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["contract"]["stage"] == "evaluate"
    assert payload["outputs"][0]["sha256"]

    changed_contract = stage_contract(
        stage="evaluate",
        command=[*command, "--device", "cuda"],
        input_paths=[config],
    )
    valid, reason = validate_stage_receipt(
        receipt,
        contract=changed_contract,
        output_paths=[output],
    )
    assert valid is False
    assert reason == "stage contract changed"


def test_study_command_forwards_resume_and_forced_stages(
    tmp_path: Path,
) -> None:
    command = build_pipeline_command(
        config="config/parameters.yaml",
        dataset="cifar100",
        request="rose_selective",
        seed=123,
        device="cuda",
        root=tmp_path,
        resume=True,
        force_stages=["evaluate", "unlearn:h_tgsd"],
    )

    assert "--resume" in command
    assert command.count("--force-stage") == 2
    assert command[-4:] == [
        "--force-stage",
        "evaluate",
        "--force-stage",
        "unlearn:h_tgsd",
    ]
