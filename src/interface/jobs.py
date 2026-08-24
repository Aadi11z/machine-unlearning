"""Job lifecycle management for unlearning requests."""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from .catalog import ArtifactCatalog, validate_job_spec


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class JobRecord:
    job_id: str
    class_id: int
    class_name: str
    request_name: str
    superclass: str
    sibling_classes: list[int]
    method: str
    steps: int
    status: JobStatus
    error: str | None = None
    checkpoint_path: str | None = None
    comparison_model: str | None = None
    candidate_id: str | None = None
    source: str | None = None
    wall_time_s: float | None = None
    created_at: float = field(default_factory=time.time)

    def public_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "request_name": self.request_name,
            "superclass": self.superclass,
            "sibling_classes": self.sibling_classes,
            "method": self.method,
            "steps": self.steps,
            "status": self.status.value,
            "error": self.error,
            "checkpoint_path": self.checkpoint_path,
            "candidate_id": self.candidate_id,
            "source": self.source,
            "wall_time_s": self.wall_time_s,
        }


JobRunner = Callable[[JobRecord], None]


class SubprocessJobRunner:
    """Run scripts/unlearn_job.py in a child process for one queued job."""

    def __init__(
        self,
        *,
        repo_root: Path,
        baseline_checkpoint: Path,
        data_dir: Path,
        output_root: Path,
        device: str = "auto",
        config_path: Path | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.baseline_checkpoint = baseline_checkpoint
        self.data_dir = data_dir
        self.output_root = output_root
        self.device = device
        self.config_path = config_path or (repo_root / "config" / "parameters.yaml")

    def __call__(self, job: JobRecord) -> None:
        output_dir = (
            self.output_root
            / "cifar100"
            / "jobs"
            / f"{job.request_name}_{job.method}_{job.steps}"
        )
        command = [
            sys.executable,
            str(self.repo_root / "scripts" / "unlearn_job.py"),
            "--forget-class",
            str(job.class_id),
            "--method",
            job.method,
            "--steps",
            str(job.steps),
            "--device",
            self.device,
            "--data-dir",
            str(self.data_dir),
            "--baseline-checkpoint",
            str(self.baseline_checkpoint),
            "--output-dir",
            str(output_dir),
            "--config",
            str(self.config_path),
        ]
        started = time.time()
        completed = subprocess.run(
            command,
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"unlearn_job failed (exit {completed.returncode}): "
                f"{completed.stderr[-2000:]}"
            )
        result_path = output_dir / "job_result.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        checkpoint = payload.get("result", {}).get("checkpoint")
        if not checkpoint:
            raise RuntimeError(f"Job result at {result_path} has no checkpoint")
        job.checkpoint_path = str(checkpoint)
        job.candidate_id = f"{job.request_name}_{job.method}_{job.steps}"
        job.comparison_model = job.candidate_id
        job.source = "job"
        job.wall_time_s = time.time() - started


class ModalJobRunner:
    """Dispatch a queued job to the HMAC-gated Modal worker endpoint."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        secret: str,
        output_root: Path,
        timeout_s: int = 4000,
        request_timeout_s: int = 30,
        poll_interval_s: float = 2.0,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.secret = secret
        self.output_root = output_root
        self.timeout_s = timeout_s
        self.request_timeout_s = request_timeout_s
        self.poll_interval_s = poll_interval_s

    def __call__(self, job: JobRecord) -> None:
        from .remote import (
            decode_checkpoint,
            parse_job_response,
            write_job_artifacts,
        )

        body = json.dumps(
            {
                "class_id": job.class_id,
                "method": job.method,
                "steps": job.steps,
            }
        ).encode("utf-8")
        started = time.time()
        status, response_body = self._post(body)
        response_payload = json.loads(response_body)
        if status == 200 and "checkpoint_b64" in response_payload:
            payload = parse_job_response(response_body)
        elif status == 202:
            call_id = str(response_payload.get("call_id", ""))
            if not call_id:
                raise RuntimeError("Worker accepted the job without a call id")
            deadline = started + self.timeout_s
            while True:
                if time.time() >= deadline:
                    raise TimeoutError(
                        f"Modal job {call_id} did not finish within {self.timeout_s}s"
                    )
                time.sleep(self.poll_interval_s)
                poll_body = json.dumps({"call_id": call_id}).encode("utf-8")
                poll_status, poll_response = self._post(poll_body)
                if poll_status == 202:
                    continue
                payload = parse_job_response(poll_response)
                break
        else:
            raise RuntimeError(f"Unexpected worker status {status}")

        _validate_worker_identity(payload, job)
        checkpoint_path = write_job_artifacts(
            output_root=self.output_root,
            class_id=job.class_id,
            class_name=job.class_name,
            request_name=job.request_name,
            superclass=job.superclass,
            sibling_classes=job.sibling_classes,
            method=job.method,
            steps=job.steps,
            checkpoint_bytes=decode_checkpoint(payload["checkpoint_b64"]),
            metrics=dict(payload.get("metrics", {})),
            wall_time_s=float(payload.get("wall_time_s", 0.0)),
        )
        job.checkpoint_path = str(checkpoint_path)
        job.candidate_id = f"{job.request_name}_{job.method}_{job.steps}"
        job.comparison_model = job.candidate_id
        job.source = "modal"
        job.wall_time_s = time.time() - started

    def _post(self, body: bytes) -> tuple[int, bytes]:
        import urllib.error
        import urllib.request

        from .remote import SIGNATURE_HEADER, sign_payload

        request = urllib.request.Request(
            self.endpoint_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                SIGNATURE_HEADER: sign_payload(self.secret, body),
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.request_timeout_s
            ) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Worker returned {error.code}: {detail}") from error


def _validate_worker_identity(payload: dict, job: JobRecord) -> None:
    """Ensure a response belongs to this request before persisting its delta."""
    expected = {
        "class_id": job.class_id,
        "class_name": job.class_name,
        "request_name": job.request_name,
        "superclass": job.superclass,
        "sibling_classes": job.sibling_classes,
        "method": job.method,
        "steps": job.steps,
    }
    try:
        actual = {
            "class_id": int(payload["class_id"]),
            "class_name": str(payload["class_name"]),
            "request_name": str(payload["request_name"]),
            "superclass": str(payload["superclass"]),
            "sibling_classes": [int(value) for value in payload["sibling_classes"]],
            "method": str(payload["method"]),
            "steps": int(payload["steps"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Worker response has invalid identity fields") from error

    mismatches = [
        field for field, expected_value in expected.items()
        if actual[field] != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "Worker response does not match the submitted job: "
            + ", ".join(mismatches)
        )


class JobManager:
    """Single-worker job queue; precomputed artifacts complete instantly."""

    def __init__(
        self,
        catalog: ArtifactCatalog,
        *,
        runner: JobRunner | None = None,
        max_completed: int = 200,
        usage_tracker=None,
        output_root: Path | None = None,
        ttl_days: float = 7.0,
    ) -> None:
        self.catalog = catalog
        self._runner = runner
        self._jobs: OrderedDict[str, JobRecord] = OrderedDict()
        self._max_completed = max_completed
        self._usage_tracker = usage_tracker
        self._output_root = output_root
        self._ttl_days = ttl_days
        self._last_cleanup = 0.0
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        if runner is not None:
            self._worker = threading.Thread(
                target=self._worker_loop, name="unlearn-job-worker", daemon=True
            )
            self._worker.start()

    def _maybe_cleanup(self) -> None:
        if self._output_root is None or time.time() - self._last_cleanup < 3600:
            return
        self._last_cleanup = time.time()
        try:
            from .guards import cleanup_expired_jobs

            cleanup_expired_jobs(self._output_root, max_age_days=self._ttl_days)
        except OSError:
            pass

    def submit(self, *, class_ref: int | str, method: str, steps: int) -> JobRecord:
        request = self.catalog.request_for(class_ref)
        class_id, method, steps = validate_job_spec(
            class_id=request.class_id, method=method, steps=steps
        )
        self._maybe_cleanup()
        job = JobRecord(
            job_id=uuid.uuid4().hex[:12],
            class_id=class_id,
            class_name=request.class_name,
            request_name=request.request_name,
            superclass=request.superclass,
            sibling_classes=list(request.sibling_classes),
            method=method,
            steps=steps,
            status=JobStatus.QUEUED,
        )
        precomputed = self.catalog.precomputed_candidate(
            class_id=class_id, method=method, steps=steps
        )
        if precomputed is not None:
            job.status = JobStatus.DONE
            job.checkpoint_path = str(precomputed.checkpoint_path)
            job.candidate_id = precomputed.candidate_id
            job.comparison_model = precomputed.comparison_model
            job.source = "precomputed"
            job.wall_time_s = 0.0
        else:
            if self._runner is None:
                raise RuntimeError(
                    "No precomputed artifact exists for this request and no "
                    "remote executor is configured"
                )
            if self._usage_tracker is not None and not self._usage_tracker.try_consume():
                raise RuntimeError(
                    "Monthly remote-job budget is exhausted; precomputed "
                    "checkpoints remain available"
                )
            job.status = JobStatus.QUEUED
        with self._lock:
            self._jobs[job.job_id] = job
            while len(self._jobs) > self._max_completed:
                self._jobs.popitem(last=False)
        if job.status is JobStatus.QUEUED:
            self._queue.put(job.job_id)
        return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def find_completed(self, candidate_id: str) -> JobRecord | None:
        with self._lock:
            for job in reversed(self._jobs.values()):
                if job.candidate_id == candidate_id and job.checkpoint_path:
                    return job
        return None

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None or job.status is not JobStatus.QUEUED:
                continue
            job.status = JobStatus.RUNNING
            try:
                assert self._runner is not None
                self._runner(job)
                job.status = JobStatus.DONE
            except Exception as error:
                job.status = JobStatus.FAILED
                job.error = str(error)
