"""FastAPI + HTMX interface for on-demand CIFAR-100 unlearning."""
from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from starlette import requests as starlette_requests

from .catalog import (
    ALLOWED_METHODS,
    BACKBONE_NAME,
    DATASET_NAME,
    NUM_CLASSES,
    ArtifactCatalog,
)
from .jobs import JobManager, JobRecord
from .probe_logic import (
    evaluate_comparison,
    find_metric_row,
    metrics_table_rows,
    top_k_rows,
)

MAX_IMAGE_PIXELS = 16_000_000
MAX_IMAGE_UPLOAD_BYTES = 10 * 1024 * 1024
_MAX_MULTIPART_OVERHEAD_BYTES = 64 * 1024
_UPLOAD_READ_CHUNK_BYTES = 64 * 1024
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]

ProbeRegistryLoader = Callable[..., object]


class UploadTooLargeError(ValueError):
    """Raised before an uploaded image can occupy unbounded application memory."""


def _upload_size_label(max_bytes: int) -> str:
    return f"{(max_bytes + 1024 * 1024 - 1) // (1024 * 1024)} MiB"


async def read_upload_limited(
    upload: UploadFile, *, max_bytes: int = MAX_IMAGE_UPLOAD_BYTES
) -> bytes:
    """Read an image in bounded chunks, rejecting it once it exceeds ``max_bytes``."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(min(_UPLOAD_READ_CHUNK_BYTES, max_bytes - total + 1))
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > max_bytes:
            raise UploadTooLargeError(
                f"Uploaded image must be at most {_upload_size_label(max_bytes)}"
            )
        chunks.append(chunk)


class ProbeService:
    """Lazy ModelRegistry wrapper producing full comparison payloads."""

    def __init__(self, registry_loader: ProbeRegistryLoader) -> None:
        self._registry_loader = registry_loader
        self._registry = None
        self._lock = threading.Lock()

    def _get(self):
        with self._lock:
            if self._registry is None:
                self._registry = self._registry_loader()
            return self._registry

    @property
    def loaded(self) -> bool:
        return self._registry is not None

    def compare(
        self,
        image: Image.Image,
        *,
        candidate,
        target_class_name: str,
        retained_label: str | None,
        top_k: int,
        comparison_rows: list[dict[str, str]],
    ) -> dict:
        registry = self._get()
        result = registry.compare(
            image,
            candidate_name=candidate.candidate_id,
            candidate_checkpoint_path=str(candidate.checkpoint_path),
            target_class_name=target_class_name,
            top_k=top_k,
        )
        metric_row = find_metric_row(comparison_rows, candidate.comparison_model)
        verdict = evaluate_comparison(
            target_class_name=target_class_name,
            candidate_name=candidate.candidate_id,
            baseline_prediction=result["baseline_prediction"],
            candidate_prediction=result["candidate_prediction"],
            metric_row=metric_row,
            retained_label=retained_label,
            num_classes=NUM_CLASSES,
        )
        return {
            "verdict": verdict,
            "candidate": {
                "candidate_id": candidate.candidate_id,
                "class_name": candidate.class_name,
                "method": candidate.method,
                "steps": candidate.steps,
                "source": candidate.source,
                "comparison_model": candidate.comparison_model,
            },
            "target_class_name": target_class_name,
            "retained_label": retained_label,
            "baseline_top_k": top_k_rows(result["baseline_prediction"]),
            "candidate_top_k": top_k_rows(result["candidate_prediction"]),
            "metrics_table": metrics_table_rows(metric_row),
        }


def decode_upload(data: bytes) -> Image.Image:
    if not data:
        raise ValueError("No image was uploaded")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as error:
        raise ValueError("The uploaded file is not a decoded image") from error
    width, height = image.size
    if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image dimensions must contain between 1 and {MAX_IMAGE_PIXELS:,} pixels"
        )
    return image.convert("RGB")


def create_app(
    *,
    catalog: ArtifactCatalog | None = None,
    job_manager: JobManager | None = None,
    probe_service: ProbeService | None = None,
    output_root: Path | None = None,
    registry_loader: ProbeRegistryLoader | None = None,
    jobs_per_hour: int | None = None,
    probes_per_minute: int | None = None,
    max_probe_upload_bytes: int = MAX_IMAGE_UPLOAD_BYTES,
) -> FastAPI:
    import os

    from .guards import SlidingWindowLimiter

    catalog = catalog or ArtifactCatalog(output_root=output_root or REPO_ROOT / "outputs")
    job_manager = job_manager or JobManager(catalog)
    probe_service = probe_service or (
        ProbeService(registry_loader) if registry_loader is not None else None
    )
    templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))
    job_limiter = SlidingWindowLimiter(
        max_events=jobs_per_hour
        or int(os.environ.get("UNML_JOBS_PER_HOUR", "6")),
        window_s=3600,
    )
    probe_limiter = SlidingWindowLimiter(
        max_events=probes_per_minute
        or int(os.environ.get("UNML_PROBES_PER_MINUTE", "60")),
        window_s=60,
    )

    if max_probe_upload_bytes < 1:
        raise ValueError("max_probe_upload_bytes must be positive")

    app = FastAPI(title="UN-ML Interface", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def limit_probe_request_size(request, call_next):
        """Reject oversized probe bodies before multipart parsing spools them."""
        if request.url.path not in {"/probe", "/api/probe"}:
            return await call_next(request)

        max_request_bytes = max_probe_upload_bytes + _MAX_MULTIPART_OVERHEAD_BYTES
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_bytes = int(content_length)
                if declared_bytes < 0:
                    raise ValueError
                if declared_bytes > max_request_bytes:
                    return PlainTextResponse(
                        f"Request body must be at most {_upload_size_label(max_probe_upload_bytes)}",
                        status_code=413,
                    )
            except ValueError:
                return PlainTextResponse("Invalid Content-Length", status_code=400)

        received = 0
        original_receive = request._receive

        async def limited_receive():
            nonlocal received
            message = await original_receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > max_request_bytes:
                    raise UploadTooLargeError(
                        f"Request body must be at most {_upload_size_label(max_probe_upload_bytes)}"
                    )
            return message

        request._receive = limited_receive
        try:
            return await call_next(request)
        except UploadTooLargeError as error:
            return PlainTextResponse(str(error), status_code=413)

    app.state.catalog = catalog
    app.state.jobs = job_manager
    app.state.probe = probe_service
    app.mount(
        "/static",
        StaticFiles(directory=str(PACKAGE_DIR / "static")),
        name="static",
    )

    def _job_context(job: JobRecord) -> dict:
        return {"job": job.public_dict()}

    def _client_key(request: starlette_requests.Request) -> str:
        return request.client.host if request.client else "unknown"

    @app.get("/", response_class=HTMLResponse)
    async def index(request: starlette_requests.Request):
        groups = _superclass_groups()
        baseline = catalog.baseline_checkpoint()
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "groups": groups,
                "methods": [m for m in ALLOWED_METHODS],
                "step_options": [50, 100, 200, 500],
                "num_classes": NUM_CLASSES,
                "backbone": BACKBONE_NAME,
                "baseline_ready": baseline is not None,
            },
        )

    @app.post("/unlearn", response_class=HTMLResponse)
    async def submit_unlearn_html(request: starlette_requests.Request):
        if not job_limiter.allow(_client_key(request)):
            return templates.TemplateResponse(
                request,
                "job_status.html",
                {
                    "job": None,
                    "error": "Too many unlearning jobs from this address; "
                    "try again later.",
                },
                status_code=429,
            )
        form = await request.form()
        try:
            job = job_manager.submit(
                class_ref=str(form.get("class_id", "")),
                method=str(form.get("method", "")),
                steps=int(str(form.get("steps", ""))),
            )
        except ValueError as error:
            return templates.TemplateResponse(
                request,
                "job_status.html",
                {"job": None, "error": str(error)},
                status_code=400,
            )
        except RuntimeError as error:
            return templates.TemplateResponse(
                request,
                "job_status.html",
                {"job": None, "error": str(error)},
                status_code=503,
            )
        return templates.TemplateResponse(
            request, "job_status.html", _job_context(job)
        )

    @app.get("/jobs/{job_id}/status", response_class=HTMLResponse)
    async def job_status_fragment(request: starlette_requests.Request, job_id: str):
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id")
        return templates.TemplateResponse(
            request, "job_status.html", _job_context(job)
        )

    @app.post("/probe", response_class=HTMLResponse)
    async def probe_html(request: starlette_requests.Request):
        if not probe_limiter.allow(_client_key(request)):
            return templates.TemplateResponse(
                request,
                "probe_result.html",
                {
                    "error": "Too many probe requests from this address; "
                    "try again in a minute.",
                    "result": None,
                },
                status_code=429,
            )
        form = await request.form()
        upload = form.get("image")
        try:
            data = (
                await read_upload_limited(upload, max_bytes=max_probe_upload_bytes)
                if hasattr(upload, "read")
                else b""
            )
            payload = _run_probe(
                image_bytes=data,
                candidate_id=str(form.get("candidate_id", "")),
                retained_label=str(form.get("retained_label") or "") or None,
            )
        except UploadTooLargeError as error:
            return templates.TemplateResponse(
                request,
                "probe_result.html",
                {"error": str(error), "result": None},
                status_code=413,
            )
        except HTTPException:
            raise
        except (ValueError, FileNotFoundError, RuntimeError) as error:
            return templates.TemplateResponse(
                request,
                "probe_result.html",
                {"error": str(error), "result": None},
            )
        return templates.TemplateResponse(
            request, "probe_result.html", {"error": None, "result": payload}
        )

    @app.post("/api/unlearn")
    async def submit_unlearn_api(
        request: starlette_requests.Request,
        class_ref: str = Form(...),
        method: str = Form(...),
        steps: int = Form(...),
    ):
        if not job_limiter.allow(_client_key(request)):
            return JSONResponse(
                {"detail": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": str(int(job_limiter.retry_after_s(_client_key(request))) or 3600)},
            )
        try:
            job = job_manager.submit(class_ref=class_ref, method=method, steps=steps)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return JSONResponse(job.public_dict(), status_code=202)

    @app.get("/api/jobs/{job_id}")
    async def job_api(job_id: str):
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id")
        return JSONResponse(job.public_dict())

    @app.post("/api/probe")
    async def probe_api(
        request: starlette_requests.Request,
        image: UploadFile = File(...),
        candidate_id: str = Form(...),
        retained_label: str = Form(""),
    ):
        if not probe_limiter.allow(_client_key(request)):
            return JSONResponse(
                {"detail": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": "60"},
            )
        try:
            data = await read_upload_limited(image, max_bytes=max_probe_upload_bytes)
            payload = _run_probe(
                image_bytes=data,
                candidate_id=candidate_id,
                retained_label=retained_label or None,
            )
        except UploadTooLargeError as error:
            raise HTTPException(status_code=413, detail=str(error)) from error
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (ValueError, FileNotFoundError, RuntimeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return JSONResponse(payload)

    @app.get("/healthz")
    async def healthz():
        return {
            "ok": True,
            "model_loaded": probe_service.loaded if probe_service else False,
        }

    def _run_probe(
        *, image_bytes: bytes, candidate_id: str, retained_label: str | None
    ) -> dict:
        if probe_service is None:
            raise RuntimeError("Probe backend is not configured")
        artifact = _artifact_for_candidate(candidate_id)
        image = decode_upload(image_bytes)
        return probe_service.compare(
            image,
            candidate=artifact,
            target_class_name=artifact.class_name,
            retained_label=retained_label,
            top_k=5,
            comparison_rows=catalog.comparison_rows(artifact.request_name),
        )

    def _artifact_for_candidate(candidate_id: str):
        job = job_manager.find_completed(candidate_id)
        if job is None:
            artifact = catalog.persisted_candidate(candidate_id)
            if artifact is None:
                raise KeyError(
                    f"Unknown or incomplete candidate {candidate_id!r}"
                )
            return artifact
        from .catalog import CandidateArtifact

        return CandidateArtifact(
            candidate_id=candidate_id,
            class_name=job.class_name,
            request_name=job.request_name,
            method=job.method,
            steps=job.steps,
            checkpoint_path=Path(job.checkpoint_path),
            comparison_model=job.comparison_model or job.method,
            source=job.source or "job",
        )

    def _superclass_groups() -> list[dict]:
        from unml.request_factory import list_superclass_groups

        return list_superclass_groups(DATASET_NAME)

    return app
