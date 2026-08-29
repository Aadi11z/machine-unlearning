from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest


def test_modal_worker_bootstraps_deployed_source_paths() -> None:
    pytest.importorskip("modal")

    from worker import modal_app

    assert "/app/src" in sys.path
    assert "/app/scripts" in sys.path
    assert modal_app.SOURCE_PATHS[-2:] == (
        modal_app.Path("/app/src"),
        modal_app.Path("/app/scripts"),
    )


def test_modal_worker_uses_the_published_secret_configuration() -> None:
    pytest.importorskip("modal")

    from worker import modal_app

    assert modal_app.JOB_SECRET_NAME == "unml-secret"
    assert modal_app.JOB_SECRET_ENV == "UNML_SECRET_KEY"


def test_modal_endpoint_spawns_and_polls_gpu_job() -> None:
    pytest.importorskip("modal")

    from worker import modal_app

    source = inspect.getsource(modal_app.job_endpoint.get_raw_f())
    assert "@app.function(timeout=ENDPOINT_TIMEOUT_S)" in source
    assert "await run_unlearn_job.spawn.aio(spec)" in source
    assert "modal.FunctionCall.from_id(call_id)" in source
    assert "await function_call.get.aio(timeout=0)" in source
    assert "await run_unlearn_job.remote.aio(spec)" not in source
    assert modal_app.ENDPOINT_TIMEOUT_S > modal_app.GPU_TIMEOUT_S
    assert modal_app.GPU_TIMEOUT_S > 300


def test_modal_worker_prepares_and_commits_shared_assets() -> None:
    pytest.importorskip("modal")

    from worker import modal_app

    prepare_source = inspect.getsource(modal_app.prepare_assets.get_raw_f())
    worker_source = inspect.getsource(modal_app.run_unlearn_job.get_raw_f())
    assert "prepare_selective_splits(" in prepare_source
    assert "data_volume.commit()" in prepare_source
    assert "hf_volume.commit()" in prepare_source
    assert "build_selective_split" not in worker_source
    assert "Prepared split missing" in worker_source
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in worker_source
    assert 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in worker_source


def test_modal_worker_caps_public_job_batch_size_for_t4_memory() -> None:
    pytest.importorskip("modal")

    from worker import modal_app

    worker_source = inspect.getsource(modal_app.run_unlearn_job.get_raw_f())
    assert modal_app.PUBLIC_JOB_BATCH_SIZE == 16
    assert "batch_size=PUBLIC_JOB_BATCH_SIZE" in worker_source


@pytest.fixture
def modal_worker():
    pytest.importorskip("modal")

    from worker import modal_app

    return modal_app


def _write_canonical_baseline(
    modal_app,
    tmp_path: Path,
    *,
    baseline_id: str | None = None,
    checkpoint_prompt_digest: str | None = None,
    include_checkpoint_prompt_contract: bool = True,
    manifest_prompt_contract: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    import torch

    from unml.baseline import resolve_baseline_paths
    from unml.manifest import build_baseline_manifest, write_baseline_manifest
    from unml.prompts import resolve_prompt_contract

    paths = resolve_baseline_paths(
        tmp_path,
        baseline_id=modal_app.CANONICAL_BASELINE_ID,
    )
    checkpoint = paths.final_fit / "promoted_adapter.pt"
    checkpoint.parent.mkdir(parents=True)
    extra = {}
    canonical_prompt = resolve_prompt_contract("cifar100")
    if include_checkpoint_prompt_contract:
        extra["prompt_contract"] = {
            "digest": checkpoint_prompt_digest or canonical_prompt.digest
        }
    prompt_contract = manifest_prompt_contract or {
        "version": canonical_prompt.version,
        "digest": canonical_prompt.digest,
        "template_count": len(canonical_prompt.templates),
    }
    torch.save({"model_config": {}, "extra": extra}, checkpoint)
    manifest = build_baseline_manifest(
        baseline_id=baseline_id or modal_app.CANONICAL_BASELINE_ID,
        dataset="cifar100",
        split={"split_id": "split-v1", "digest": "split-digest"},
        model_config={},
        prompt_contract=prompt_contract,
        checkpoints={"checkpoint": checkpoint},
        metrics={},
    )
    manifest_path = paths.final_fit / modal_app.CANONICAL_BASELINE_MANIFEST_NAME
    write_baseline_manifest(manifest_path, manifest)
    return checkpoint, manifest_path


def test_modal_worker_selects_only_the_explicit_verified_canonical_baseline(
    modal_worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, _ = _write_canonical_baseline(modal_worker, tmp_path)
    legacy = tmp_path / "baseline_0000" / "checkpoints" / "finetuned_best.pt"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy")
    monkeypatch.setattr(modal_worker, "ARTIFACT_ROOT", tmp_path)

    assert modal_worker._find_baseline() == checkpoint


def test_modal_worker_rejects_missing_canonical_manifest(
    modal_worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(modal_worker, "ARTIFACT_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="manifest missing"):
        modal_worker._find_baseline()


def test_modal_worker_rejects_wrong_canonical_baseline_identity(
    modal_worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_canonical_baseline(modal_worker, tmp_path, baseline_id="other-baseline")
    monkeypatch.setattr(modal_worker, "ARTIFACT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="Baseline id mismatch"):
        modal_worker._find_baseline()


def test_modal_worker_rejects_tampered_canonical_checkpoint(
    modal_worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, _ = _write_canonical_baseline(modal_worker, tmp_path)
    checkpoint.write_bytes(b"tampered")
    monkeypatch.setattr(modal_worker, "ARTIFACT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="does not match"):
        modal_worker._find_baseline()


@pytest.mark.parametrize("checkpoint_path", ["../legacy.pt", "/tmp/legacy.pt"])
def test_modal_worker_rejects_checkpoint_paths_outside_the_canonical_directory(
    modal_worker,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: str,
) -> None:
    _, manifest_path = _write_canonical_baseline(modal_worker, tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["checkpoint"]["path"] = checkpoint_path
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(modal_worker, "ARTIFACT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="must be relative|escapes"):
        modal_worker._find_baseline()


@pytest.mark.parametrize(
    ("include_checkpoint_prompt_contract", "checkpoint_prompt_digest"),
    [(False, None), (True, "legacy-prompt")],
)
def test_modal_worker_rejects_missing_or_mismatched_checkpoint_prompt_contract(
    modal_worker,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_checkpoint_prompt_contract: bool,
    checkpoint_prompt_digest: str | None,
) -> None:
    _write_canonical_baseline(
        modal_worker,
        tmp_path,
        checkpoint_prompt_digest=checkpoint_prompt_digest,
        include_checkpoint_prompt_contract=include_checkpoint_prompt_contract,
    )
    monkeypatch.setattr(modal_worker, "ARTIFACT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="prompt contract does not match"):
        modal_worker._find_baseline()


def test_modal_worker_rejects_a_legacy_prompt_manifest_and_checkpoint_that_agree(
    modal_worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_canonical_baseline(
        modal_worker,
        tmp_path,
        checkpoint_prompt_digest="legacy-prompt",
        manifest_prompt_contract={
            "version": "legacy_single_template",
            "digest": "legacy-prompt",
            "template_count": 1,
        },
    )
    monkeypatch.setattr(modal_worker, "ARTIFACT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="Canonical baseline prompt contract mismatch"):
        modal_worker._find_baseline()
