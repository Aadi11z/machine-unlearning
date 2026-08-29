"""Modal worker: runs one single-class unlearning job per request.

Setup:
    uv sync --locked --group modal
    modal setup
    modal secret create unml-secret UNML_SECRET_KEY="$UNML_JOB_SECRET"
    modal volume put unml-artifacts \
      <outputs>/cifar100/canonical/cifar100_canonical_v1 \
      cifar100/canonical/cifar100_canonical_v1
    modal run worker/modal_app.py::prepare_assets
    modal deploy worker/modal_app.py

The endpoint accepts HMAC-signed job specs and returns the complete candidate
adapter state (~1.2 MB, base64 safetensors) plus summary metrics.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKER_DIR.parent
SOURCE_PATHS = (
    REPO_ROOT,
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
    Path("/app/src"),
    Path("/app/scripts"),
)
for path in map(str, SOURCE_PATHS):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    import modal
except ImportError as error:  # pragma: no cover - optional deployment dependency
    raise RuntimeError(
        "The Modal worker requires the `modal` package. "
        "Install it with: uv sync --locked --group modal"
    ) from error

from fastapi import Request  # noqa: E402

from interface.catalog import validate_job_spec  # noqa: E402
from interface.remote import SIGNATURE_HEADER, encode_checkpoint, verify_signature  # noqa: E402
from unml.baseline import CANONICAL_BASELINE_ID, resolve_baseline_paths  # noqa: E402
from unml.manifest import (  # noqa: E402
    validate_baseline_manifest,
    validate_baseline_identity,
    verify_manifest_artifacts,
)
from unml.prompts import resolve_prompt_contract  # noqa: E402

GPU_KIND = os.environ.get("UNML_MODAL_GPU", "T4")
# These must match the published Modal secret in the active workspace/environment.
JOB_SECRET_NAME = "unml-secret"
JOB_SECRET_ENV = "UNML_SECRET_KEY"
GPU_TIMEOUT_S = 3600
ENDPOINT_TIMEOUT_S = GPU_TIMEOUT_S + 300
ASSET_TIMEOUT_S = 3600
DEFAULT_SEED = 42
# GA+KL retains two student activation graphs until backward and also runs a
# teacher forward. The repository research default of 128 exhausts a 16 GiB
# T4, so public jobs use a conservative physical batch size.
PUBLIC_JOB_BATCH_SIZE = 16
DATA_DIR = Path("/vol/data")
SPLIT_ROOT = DATA_DIR / "splits" / "cifar100"
ARTIFACT_ROOT = Path("/vol/artifacts")
CANONICAL_BASELINE_MANIFEST_NAME = "manifest.json"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.40.0,<5.0",
        "safetensors>=0.4.0",
        "tqdm>=4.66.0",
        "PyYAML>=6.0",
        "numpy>=1.26.0,<2.0",
        "Pillow>=10.0",
        "fastapi>=0.115.0",
    )
    .env({"HF_HOME": "/vol/hf"})
    .add_local_dir(REPO_ROOT / "src", "/app/src")
    .add_local_dir(REPO_ROOT / "scripts", "/app/scripts")
    .add_local_dir(REPO_ROOT / "config", "/app/config")
)
data_volume = modal.Volume.from_name("unml-data", create_if_missing=True)
hf_volume = modal.Volume.from_name("unml-hf", create_if_missing=True)
artifact_volume = modal.Volume.from_name("unml-artifacts", create_if_missing=True)

app = modal.App(
    "unml-worker",
    image=image,
    volumes={
        "/vol/data": data_volume,
        "/vol/hf": hf_volume,
        "/vol/artifacts": artifact_volume,
    },
    secrets=[modal.Secret.from_name(JOB_SECRET_NAME)],
)


def _resolve_manifest_artifact_path(
    baseline_root: Path, role: str, record: object
) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"Canonical baseline artifact {role!r} is invalid")
    relative_path = record.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"Canonical baseline artifact {role!r} lacks a path")
    artifact_path = Path(relative_path)
    if artifact_path.is_absolute():
        raise ValueError(
            f"Canonical baseline artifact {role!r} path must be relative"
        )
    resolved_root = baseline_root.resolve()
    resolved_artifact = (resolved_root / artifact_path).resolve()
    try:
        resolved_artifact.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"Canonical baseline artifact {role!r} escapes its baseline directory"
        ) from error
    return resolved_artifact


def _find_baseline() -> Path:
    """Return the one promoted baseline only after provenance verification."""
    paths = resolve_baseline_paths(
        ARTIFACT_ROOT,
        baseline_id=CANONICAL_BASELINE_ID,
    )
    baseline_root = paths.final_fit
    manifest_path = baseline_root / CANONICAL_BASELINE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Canonical baseline manifest missing at "
            f"{manifest_path}. Upload {CANONICAL_BASELINE_ID} with its manifest."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Canonical baseline manifest is invalid JSON: {manifest_path}"
        ) from error
    validate_baseline_manifest(manifest)
    if manifest.get("dataset") != "cifar100":
        raise ValueError(
            "Canonical baseline manifest dataset mismatch: "
            f"expected 'cifar100', received {manifest.get('dataset')!r}"
        )
    canonical_prompt = resolve_prompt_contract("cifar100")
    expected_prompt = {
        "version": canonical_prompt.version,
        "digest": canonical_prompt.digest,
        "template_count": len(canonical_prompt.templates),
    }
    manifest_prompt = manifest["prompt_contract"]
    if not isinstance(manifest_prompt, dict) or any(
        manifest_prompt.get(key) != value for key, value in expected_prompt.items()
    ):
        raise ValueError(
            "Canonical baseline prompt contract mismatch: "
            f"expected {expected_prompt!r}, received {manifest_prompt!r}"
        )

    artifact_paths = {
        role: _resolve_manifest_artifact_path(baseline_root, role, record)
        for role, record in manifest["artifacts"].items()
    }
    checkpoint = artifact_paths.get("checkpoint") or artifact_paths.get("best")
    if checkpoint is None:
        raise ValueError("Canonical baseline manifest lacks a checkpoint artifact")
    verify_manifest_artifacts(manifest, root=baseline_root)
    validate_baseline_identity(
        manifest,
        baseline_id=CANONICAL_BASELINE_ID,
        checkpoint_path=checkpoint,
    )
    return checkpoint


def _split_path(request_name: str, seed: int) -> Path:
    return SPLIT_ROOT / f"seed_{seed}" / f"{request_name}_split.json"


@app.function(timeout=ASSET_TIMEOUT_S, memory=4096)
def prepare_assets(seed: int = DEFAULT_SEED) -> dict:
    """Download shared assets and generate every class split once on CPU."""
    from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

    from unml.config import load_runtime_config, resolve_model_value
    from unml.request_factory import prepare_selective_splits

    runtime_cfg = load_runtime_config("/app/config/parameters.yaml")
    split_paths = prepare_selective_splits(
        data_dir=str(DATA_DIR),
        split_dir=str(SPLIT_ROOT / f"seed_{seed}"),
        forget_fraction=1.0,
        retain_val_fraction=float(
            runtime_cfg.get("data", {}).get("retain_val_fraction", 0.1)
        ),
        seed=seed,
        dataset_name="cifar100",
    )
    data_volume.commit()

    model_name = resolve_model_value(
        None,
        runtime_cfg,
        "cifar100",
        "model_name",
        "openai/clip-vit-base-patch16",
    )
    CLIPImageProcessor.from_pretrained(model_name)
    CLIPTokenizer.from_pretrained(model_name)
    CLIPModel.from_pretrained(model_name)
    hf_volume.commit()
    return {
        "dataset": "cifar100",
        "model_name": model_name,
        "seed": seed,
        "split_count": len(split_paths),
    }


@app.function(gpu=GPU_KIND, timeout=GPU_TIMEOUT_S)
def run_unlearn_job(spec: dict) -> dict:
    """Execute one unlearning job in-process on a worker GPU."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import unlearn_job
    from unml.config import load_runtime_config
    from unml.request_factory import resolve_selective_request
    from unml.unlearn import run_unlearning

    class_id, method, steps = validate_job_spec(
        class_id=int(spec["class_id"]),
        method=str(spec["method"]),
        steps=int(spec["steps"]),
    )
    seed = int(spec.get("seed", 42))
    device = str(spec.get("device", "cuda"))
    request = resolve_selective_request("cifar100", class_id)

    data_volume.reload()
    hf_volume.reload()
    artifact_volume.reload()
    runtime_cfg = load_runtime_config("/app/config/parameters.yaml")
    split_path = _split_path(request.request_name, seed)
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Prepared split missing at {split_path}. Run: "
            f"modal run worker/modal_app.py::prepare_assets --seed {seed}"
        )

    cfg = unlearn_job.build_job_unlearn_config(
        runtime_cfg,
        request=request,
        dataset_name="cifar100",
        data_dir=str(DATA_DIR),
        split_path=str(split_path),
        finetuned_checkpoint=str(_find_baseline()),
        output_dir="/tmp/job_out",
        method=method,
        steps=steps,
        seed=seed,
        device=device,
        batch_size=PUBLIC_JOB_BATCH_SIZE,
    )
    started = time.time()
    result = run_unlearning(cfg)
    wall_time = time.time() - started

    from unml.model import export_checkpoint_safetensors, read_checkpoint_payload

    payload = read_checkpoint_payload(str(Path(result["checkpoint"])))
    checkpoint_bytes = export_checkpoint_safetensors(payload)
    return {
        "checkpoint_b64": encode_checkpoint(checkpoint_bytes),
        "checkpoint_format": "safetensors",
        "class_id": class_id,
        "class_name": request.class_name,
        "request_name": request.request_name,
        "superclass": request.superclass,
        "sibling_classes": list(request.sibling_classes),
        "method": method,
        "steps": steps,
        "wall_time_s": wall_time,
        "metrics": {
            key: value
            for key, value in result.items()
            if isinstance(value, (int, float, str))
        }
        | {"batch_size": PUBLIC_JOB_BATCH_SIZE},
    }


@app.function(timeout=ENDPOINT_TIMEOUT_S)
@modal.fastapi_endpoint(method="POST", label="unml-job-endpoint")
async def job_endpoint(request: Request):
    """HMAC-gated HTTP entry point used by the platform."""
    from fastapi.responses import JSONResponse

    body = await request.body()
    secret = os.environ[JOB_SECRET_ENV]
    signature = request.headers.get(SIGNATURE_HEADER)
    if not verify_signature(secret, body, signature):
        return JSONResponse({"error": "invalid signature"}, status_code=401)
    try:
        spec = json.loads(body)
        if "call_id" in spec:
            call_id = str(spec["call_id"]).strip()
            if not call_id:
                raise ValueError("call_id must be non-empty")
            function_call = modal.FunctionCall.from_id(call_id)
            try:
                result = await function_call.get.aio(timeout=0)
            except TimeoutError:
                return JSONResponse(
                    {"status": "running", "call_id": call_id}, status_code=202
                )
            except modal.exception.OutputExpiredError:
                return JSONResponse({"error": "job result expired"}, status_code=404)
            return JSONResponse(result)
        validate_job_spec(
            class_id=int(spec["class_id"]),
            method=str(spec["method"]),
            steps=int(spec["steps"]),
        )
    except (ValueError, KeyError, json.JSONDecodeError) as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    call = await run_unlearn_job.spawn.aio(spec)
    return JSONResponse(
        {"status": "accepted", "call_id": call.object_id}, status_code=202
    )
