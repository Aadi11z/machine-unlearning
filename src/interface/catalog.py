"""Artifact lookup and job-spec validation for the platform interface."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from unml.baseline import BaselineReference, configured_baseline_manifest, resolve_baseline
from unml.manifest import (
    baseline_checkpoint_record,
    sha256_file,
    validate_baseline_identity,
    verify_retraining_oracle_canonical_contract,
    verify_retraining_oracle_manifest,
)
from unml.methods import UNLEARNING_METHODS
from unml.prompts import resolve_prompt_contract
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


@dataclass(frozen=True)
class ReferenceArtifact:
    oracle_id: str
    request_name: str
    request_type: str
    forget_class_names: tuple[str, ...]
    checkpoint_path: Path
    manifest_path: Path
    metric_rows: tuple[dict[str, str], ...]


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
        baseline_manifest_path: Path | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.dataset_name = dataset_name
        self._baseline_override = baseline_checkpoint_path
        self._comparison_cache: dict[str, list[dict[str, str]]] = {}
        self._baseline_manifest_path = baseline_manifest_path
        self._baseline_reference: BaselineReference | None = None
        self._baseline_manifest_cache: dict[str, Any] | None = None
        self._reference_cache: tuple[ReferenceArtifact, ...] | None = None

    def request_for(self, class_id: int):
        return resolve_selective_request(self.dataset_name, class_id)

    def baseline_checkpoint(self) -> Path | None:
        if self._baseline_override is not None:
            return self._baseline_override
        try:
            baseline = self.baseline_reference()
        except FileNotFoundError:
            return None
        return baseline.final_checkpoint

    def baseline_reference(self) -> BaselineReference:
        if self._baseline_reference is None:
            path = self._baseline_manifest_path or configured_baseline_manifest(
                self.output_root, dataset=self.dataset_name
            )
            self._baseline_reference = resolve_baseline(path, dataset=self.dataset_name)
        return self._baseline_reference

    def baseline_identity(self) -> dict[str, str]:
        """Return the exact baseline identity used by jobs and persisted results."""
        checkpoint = self.baseline_checkpoint()
        if checkpoint is None:
            raise FileNotFoundError("No baseline checkpoint is available")
        if self._baseline_override is not None:
            return {
                "baseline_id": "legacy_override",
                "baseline_sha256": sha256_file(checkpoint),
            }
        baseline = self.baseline_reference()
        return {
            "baseline_id": baseline.baseline_id,
            "baseline_sha256": baseline.final_checkpoint_sha256,
        }

    def canonical_baseline_root(self) -> Path:
        return self.baseline_reference().package_root

    def baseline_manifest(self) -> dict[str, Any]:
        if self._baseline_manifest_cache is None:
            baseline = self.baseline_reference()
            manifest = dict(baseline.manifest)
            if manifest.get("dataset") != self.dataset_name:
                raise ValueError("Baseline dataset does not match interface")
            contract = resolve_prompt_contract(self.dataset_name)
            stored_contract = manifest.get("prompt_contract", {})
            if (
                stored_contract.get("version") != contract.version
                or stored_contract.get("digest") != contract.digest
            ):
                raise ValueError("Baseline prompt contract is not supported")
            baseline_checkpoint_record(manifest)
            validate_baseline_identity(
                manifest,
                baseline_id=baseline.baseline_id,
                checkpoint_path=baseline.final_checkpoint,
            )
            self._baseline_manifest_cache = manifest
        return self._baseline_manifest_cache

    def baseline_class_names(self) -> list[str]:
        if self._baseline_override is not None:
            from unml.data import get_dataset_spec

            return list(get_dataset_spec(self.dataset_name).class_names)
        metrics = self.baseline_manifest().get("metrics", {})
        class_names = metrics.get("class_names") if isinstance(metrics, Mapping) else None
        if not isinstance(class_names, list) or len(class_names) != NUM_CLASSES:
            raise ValueError(
                "Canonical baseline manifest must contain all CIFAR-100 class names"
            )
        return [str(name) for name in class_names]

    def reference_artifacts(self) -> tuple[ReferenceArtifact, ...]:
        if self._reference_cache is not None:
            return self._reference_cache
        root = self.output_root / self.dataset_name / "oracle"
        index_path = root / "promoted.json"
        if not index_path.is_file():
            self._reference_cache = ()
            return self._reference_cache
        index = _read_json_object(index_path)
        if index.get("schema") != "unml-retraining-oracle-index-v1" or not isinstance(
            index.get("references"), Mapping
        ):
            raise ValueError(f"Invalid retraining oracle index: {index_path}")
        manifests = []
        resolved_root = root.resolve()
        for relative_path in index["references"].values():
            manifest_path = (root / str(relative_path)).resolve()
            if not manifest_path.is_relative_to(resolved_root):
                raise ValueError("Retraining oracle index path escapes its root")
            manifests.append(manifest_path)
        references: list[ReferenceArtifact] = []
        canonical = self.baseline_manifest()
        canonical_manifest_path = self.canonical_baseline_root() / "manifest.json"
        for manifest_path in manifests:
            manifest = _read_json_object(manifest_path)
            verify_retraining_oracle_manifest(manifest, root=manifest_path.parent)
            if manifest.get("dataset") != self.dataset_name:
                raise ValueError(f"Oracle dataset mismatch: {manifest_path}")
            verify_retraining_oracle_canonical_contract(
                manifest,
                canonical,
                canonical_manifest_path=canonical_manifest_path,
            )
            request = manifest["request"]
            request_name = str(request["request_name"])
            if index["references"].get(request_name) != str(
                manifest_path.relative_to(resolved_root)
            ):
                raise ValueError("Promoted oracle request does not match its manifest")
            checkpoint = manifest_path.parent / manifest["artifacts"]["checkpoint"]["path"]
            references.append(
                ReferenceArtifact(
                    oracle_id=str(manifest["oracle_id"]),
                    request_name=request_name,
                    request_type=str(request["request_type"]),
                    forget_class_names=tuple(
                        str(name) for name in request["forget_class_names"]
                    ),
                    checkpoint_path=checkpoint,
                    manifest_path=manifest_path,
                    metric_rows=tuple(_reference_metric_rows(canonical, manifest)),
                )
            )
        self._reference_cache = tuple(references)
        return self._reference_cache

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
        # Historical demonstrations predate the canonical artifact layout.  They
        # remain probeable as explicitly precomputed candidates, but are never
        # considered a baseline or fresh, identity-bound interface job.
        historical_dirs = (
            dataset_root / "archive" / "legacy" / request.request_name,
        )
        job_checkpoint = _first_existing(
            [job_dir / name for name in _CHECKPOINT_FILENAMES(method)]
        )
        if job_checkpoint is not None and not self._job_matches_baseline(
            job_root / "job_result.json"
        ):
            job_checkpoint = None
        checkpoint = job_checkpoint
        if checkpoint is None:
            for historical_dir in historical_dirs:
                checkpoint = _first_existing(
                    sorted(
                        path
                        for path in historical_dir.glob(
                            f"unlearn_{method}*_{steps}/checkpoints/*"
                        )
                        if path.suffix in {".pt", ".safetensors"}
                    )
                )
                if checkpoint is not None:
                    break
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

    def _job_matches_baseline(self, result_path: Path) -> bool:
        if not result_path.is_file():
            return False
        try:
            payload = _read_json_object(result_path)
            expected = self.baseline_identity()
        except (FileNotFoundError, OSError, TypeError, ValueError):
            return False
        return all(payload.get(key) == value for key, value in expected.items())

    def persisted_candidate(self, candidate_id: str) -> CandidateArtifact | None:
        """Resolve a completed interface job after the web process restarts."""
        import json

        jobs_root = self.output_root / self.dataset_name / "jobs"
        if not candidate_id or not jobs_root.is_dir():
            return None

        for result_path in jobs_root.glob("*/job_result.json"):
            if not self._job_matches_baseline(result_path):
                continue
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
                [
                    path
                    for root in (
                        dataset_root / "archive" / "legacy" / request_name,
                    )
                    for path in root.glob("eval_compare_*/comparison.csv")
                ],
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
            if not self._job_matches_baseline(result_path):
                continue
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


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read artifact manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact manifest must contain an object: {path}")
    return payload


def _reference_metric_rows(
    _canonical: Mapping[str, Any], oracle: Mapping[str, Any]
) -> list[dict[str, str]]:
    evaluations = oracle["comparison"].get("evaluations", {})
    canonical = evaluations.get("canonical", {})
    retraining_oracle = evaluations.get("oracle", {})
    rows = []
    for key, label in (
        ("test_all", "Official test (100 classes)"),
        ("test_retain", "Retain test (non-target classes)"),
        ("test_forget", "Forget test (target classes)"),
    ):
        canonical_value = _metric(canonical.get(key, {}), "accuracy")
        oracle_value = _metric(retraining_oracle.get(key, {}), "accuracy")
        rows.append(_metric_row(label, canonical_value, oracle_value))
    return rows


def _metric(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if not _is_metric_number(value):
        raise ValueError(f"Recorded reference metrics lack {name!r}")
    return float(value)


def _metric_row(label: str, canonical: float | None, oracle: float | None) -> dict[str, str]:
    delta = oracle - canonical if canonical is not None and oracle is not None else None
    return {
        "label": label,
        "canonical": _percent(canonical),
        "oracle": _percent(oracle),
        "delta": f"{delta * 100:+.2f} pp" if delta is not None else "not recorded",
    }


def _percent(value: float | None) -> str:
    return f"{value * 100:.2f}%" if value is not None else "not recorded"
