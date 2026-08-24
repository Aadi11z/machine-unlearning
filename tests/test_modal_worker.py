from __future__ import annotations

import inspect
import sys

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
