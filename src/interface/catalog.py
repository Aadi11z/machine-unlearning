"""Artifact lookup and job-spec validation for the platform interface."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from unml.methods import UNLEARNING_METHODS
from unml.request_factory import resolve_selective_request

DATASET_NAME = "cifar100"
NUM_CLASSES = 100
BACKBONE_NAME = "openai/clip-vit-base-patch16"

ALLOWED_METHODS = UNLEARNING_METHODS
METHOD_TO_COMPARISON_MODEL = {
    **{method: method for method in ALLOWED_METHODS},
    "h_tgsd_no_sibling_preservation": "h_tgsd_no_sibling",
}
MIN_STEPS = 10
MAX_STEPS = 500


@dataclass(frozen=True)
class CandidateArtifact:
    candidate_id: str
    class_name: str
    request_name: str
    method: str
    steps: int
    checkpoint_path: Path
    comparison_model: str
    source: str


def validate_job_spec(
    *, class_id: int, method: str, steps: int
) -> tuple[int, str, int]:
    try:
        class_id = int(class_id)
        steps = int(steps)
    except (TypeError, ValueError) as error:
        raise ValueError("class_id and steps must be integers") from error
    if class_id < 0 or class_id >= NUM_CLASSES:
        raise ValueError(f"class_id must be within [0, {NUM_CLASSES - 1}]")
    if method not in ALLOWED_METHODS:
        raise ValueError(f"method must be one of {sorted(ALLOWED_METHODS)}")
    if steps < MIN_STEPS or steps > MAX_STEPS:
        raise ValueError(f"steps must be within [{MIN_STEPS}, {MAX_STEPS}]")
    return class_id, method, steps


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _CHECKPOINT_FILENAMES(method: str) -> tuple[Path, ...]:
    return (
        Path(f"unlearn_{method}.safetensors"),
        Path(f"unlearn_{method}.pt"),
    )


class ArtifactCatalog:
    """Resolve precomputed checkpoints, baseline, and comparison metrics."""

    def __init__(
        self,
        *,
        output_root: Path,
        dataset_name: str = DATASET_NAME,
        baseline_checkpoint_path: Path | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.dataset_name = dataset_name
        self._baseline_override = baseline_checkpoint_path
        self._comparison_cache: dict[str, list[dict[str, str]]] = {}

    def request_for(self, class_id: int):
        return resolve_selective_request(self.dataset_name, class_id)

    def baseline_checkpoint(self) -> Path | None:
        if self._baseline_override is not None:
            return self._baseline_override
        dataset_root = self.output_root / self.dataset_name
        if not dataset_root.is_dir():
            return None
        matches = sorted(dataset_root.glob("*/baseline_*/checkpoints/finetuned_best.pt"))
        return matches[0] if matches else None

    def precomputed_candidate(
        self, *, class_id: int, method: str, steps: int
    ) -> CandidateArtifact | None:
        request = self.request_for(class_id)
        dataset_root = self.output_root / self.dataset_name
        job_root = (
            dataset_root
            / "jobs"
            / f"{request.request_name}_{method}_{steps}"
        )
        job_dir = job_root / "checkpoints"
        historical_dir = dataset_root / request.request_name
        job_checkpoint = _first_existing(
            [job_dir / name for name in _CHECKPOINT_FILENAMES(method)]
        )
        checkpoint = job_checkpoint or _first_existing(
            sorted(
                path
                for path in historical_dir.glob(
                    f"unlearn_{method}*_{steps}/checkpoints/*"
                )
                if path.suffix in {".pt", ".safetensors"}
            )
        )
        if checkpoint is None:
            return None
        candidate_id = f"{request.request_name}_{method}_{steps}"
        is_fresh_job = job_checkpoint is not None and (job_root / "job_result.json").is_file()
        return CandidateArtifact(
            candidate_id=candidate_id,
            class_name=request.class_name,
            request_name=request.request_name,
            method=method,
            steps=steps,
            checkpoint_path=checkpoint,
            comparison_model=(candidate_id if is_fresh_job else METHOD_TO_COMPARISON_MODEL[method]),
            source="job" if is_fresh_job else "precomputed",
        )

    def persisted_candidate(self, candidate_id: str) -> CandidateArtifact | None:
        """Resolve a completed interface job after the web process restarts."""
        import json

        jobs_root = self.output_root / self.dataset_name / "jobs"
        if not candidate_id or not jobs_root.is_dir():
            return None

        for result_path in jobs_root.glob("*/job_result.json"):
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                stored_candidate_id = str(
                    payload.get("candidate_id", result_path.parent.name)
                )
                if stored_candidate_id != candidate_id:
                    continue
                class_id, method, steps = validate_job_spec(
                    class_id=int(payload["class_id"]),
                    method=str(payload["method"]),
                    steps=int(payload["steps"]),
                )
                request = self.request_for(class_id)
                actual_identity = {
                    "class_name": str(payload["class_name"]),
                    "request_name": str(payload["request_name"]),
                    "superclass": str(payload["superclass"]),
                    "sibling_classes": [
                        int(value) for value in payload["sibling_classes"]
                    ],
                }
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

            expected_identity = {
                "class_name": request.class_name,
                "request_name": request.request_name,
                "superclass": request.superclass,
                "sibling_classes": list(request.sibling_classes),
            }
            if actual_identity != expected_identity:
                continue
            candidate = self.precomputed_candidate(
                class_id=class_id,
                method=method,
                steps=steps,
            )
            if (
                candidate is not None
                and candidate.candidate_id == candidate_id
                and candidate.source == "job"
            ):
                return candidate
        return None

    def comparison_rows(self, request_name: str) -> list[dict[str, str]]:
        if request_name not in self._comparison_cache:
            rows: list[dict[str, str]] = []
            dataset_root = self.output_root / self.dataset_name
            candidates = sorted(
                (dataset_root / request_name).glob("eval_compare_*/comparison.csv"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                import csv

                with candidates[0].open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
            self._comparison_cache[request_name] = rows
        # A remote job can finish after the first probe for this request, so
        # job-result rows must be discovered on every lookup.
        return [
            *self._comparison_cache[request_name],
            *self._fresh_job_metric_rows(request_name),
        ]

    def _fresh_job_metric_rows(self, request_name: str) -> list[dict[str, str]]:
        """Translate the quick post-job evaluation into the UI metric schema.

        Fresh jobs evaluate the forgotten class, all retained classes, and all
        classes.  They do not separately evaluate sibling and unrelated classes,
        so those fields are intentionally absent rather than inferred.
        """
        import json

        jobs_root = self.output_root / self.dataset_name / "jobs"
        if not jobs_root.is_dir():
            return []

        rows: list[dict[str, str]] = []
        for result_path in jobs_root.glob(f"{request_name}_*/job_result.json"):
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                if str(payload["request_name"]) != request_name:
                    continue
                candidate_id = str(payload.get("candidate_id", result_path.parent.name))
                result = payload["result"]
                metrics = result.get("metrics", result)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(metrics, dict):
                continue
            row = {"model": candidate_id}
            for source_key, display_key in (
                ("target_test_acc", "target_test_acc"),
                ("test_retain_acc", "retained_test_acc"),
                ("test_all_acc", "utility_test_all"),
            ):
                value = metrics.get(source_key)
                if _is_metric_number(value):
                    row[display_key] = str(value)
            if len(row) > 1:
                rows.append(row)
        return rows


def _is_metric_number(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True
