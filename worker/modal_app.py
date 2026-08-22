"""Modal worker: runs one single-class unlearning job per request.

Setup:
    uv pip install modal
    modal secret create unml-job-secret UNML_JOB_SECRET=<hex>
    modal volume put unml-artifacts <path-to>/baseline_2000 baseline_2000
    modal deploy worker/modal_app.py

The endpoint accepts HMAC-signed job specs and returns the LoRA delta
(~1.2 MB, base64) plus summary metrics in one JSON response.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKER_DIR.parent
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    import modal
except ImportError as error:  # pragma: no cover - optional deployment dependency
    raise RuntimeError(
        "The Modal worker requires the `modal` package. "
        "Install it with: uv pip install modal"
    ) from error

from interface.catalog import validate_job_spec  # noqa: E402
from interface.remote import SIGNATURE_HEADER, encode_checkpoint, verify_signature  # noqa: E402

GPU_KIND = os.environ.get("UNML_MODAL_GPU", "T4")
JOB_SECRET_NAME = "unml-job-secret"
TIMEOUT_S = 3600

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.40.0,<5.0",
        "tqdm>=4.66.0",
        "PyYAML>=6.0",
        "numpy>=1.26.0,<2.0",
        "pandas>=2.2.0",
        "tabulate>=0.9.0",
        "scikit-learn>=1.4.0",
        "matplotlib>=3.8.0",
        "Pillow>=10.0",
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


def _find_baseline() -> Path:
    matches = sorted(
        Path("/vol/artifacts").glob("baseline_*/checkpoints/finetuned_best.pt")
    )
    if not matches:
        raise FileNotFoundError(
            "Upload the class-agnostic baseline first: "
            "modal volume put unml-artifacts <outputs>/cifar100/rose_selective/baseline_2000 baseline_2000"
        )
    return matches[0]


@app.function(gpu=GPU_KIND, timeout=TIMEOUT_S)
def run_unlearn_job(spec: dict) -> dict:
    """Execute one unlearning job in-process on a worker GPU."""
    import unlearn_job
    from unml.config import load_runtime_config
    from unml.request_factory import (
        build_selective_split,
        resolve_selective_request,
    )
    from unml.unlearn import run_unlearning

    class_id, method, steps = validate_job_spec(
        class_id=int(spec["class_id"]),
        method=str(spec["method"]),
        steps=int(spec["steps"]),
    )
    seed = int(spec.get("seed", 42))
    device = str(spec.get("device", "cuda"))
    request = resolve_selective_request("cifar100", class_id)

    runtime_cfg = load_runtime_config("/app/config/parameters.yaml")
    split_path = Path("/tmp") / f"{request.request_name}_split.json"
    if not split_path.is_file():
        build_selective_split(
            data_dir="/vol/data",
            split_path=str(split_path),
            request=request,
            forget_fraction=1.0,
            retain_val_fraction=float(
                runtime_cfg.get("data", {}).get("retain_val_fraction", 0.1)
            ),
            seed=seed,
            dataset_name="cifar100",
        )

    cfg = unlearn_job.build_job_unlearn_config(
        runtime_cfg,
        request=request,
        dataset_name="cifar100",
        data_dir="/vol/data",
        split_path=str(split_path),
        finetuned_checkpoint=str(_find_baseline()),
        output_dir="/tmp/job_out",
        method=method,
        steps=steps,
        seed=seed,
        device=device,
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
        },
    }


@app.fastapi_endpoint(method="POST", label="unml-job-endpoint")
async def job_endpoint(request):
    """HMAC-gated HTTP entry point used by the platform."""
    from fastapi.responses import JSONResponse

    body = await request.body()
    secret = os.environ["UNML_JOB_SECRET"]
    signature = request.headers.get(SIGNATURE_HEADER)
    if not verify_signature(secret, body, signature):
        return JSONResponse({"error": "invalid signature"}, status_code=401)
    try:
        spec = json.loads(body)
        validate_job_spec(
            class_id=int(spec["class_id"]),
            method=str(spec["method"]),
            steps=int(spec["steps"]),
        )
    except (ValueError, KeyError, json.JSONDecodeError) as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    result = run_unlearn_job.remote(spec)
    return JSONResponse(result)
