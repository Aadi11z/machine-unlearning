from __future__ import annotations

import gc
import html
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from .config import load_runtime_config, resolve_output_root
from .data import load_split_metadata
from .probe import (
    ImageCheckpointPredictor,
    load_image_checkpoint_predictor,
    predict_probe_image,
)


MAX_IMAGE_PIXELS = 16_000_000
_METRIC_COLUMNS = (
    ("target_test_acc", "Target test accuracy"),
    ("sibling_test_acc", "Sibling test accuracy"),
    ("unrelated_test_acc", "Unrelated test accuracy"),
    ("utility_test_all", "Overall test accuracy"),
    ("mia_auc_confidence", "Confidence MIA AUC"),
    ("mia_auc_loss", "Loss MIA AUC"),
)

_APP_CSS = """
.gradio-container {
  max-width: 1240px !important;
  margin: 0 auto !important;
  padding: 28px 24px 48px !important;
}
.probe-header {
  border-left: 4px solid #f97316;
  padding: 4px 0 4px 18px;
  margin: 8px 0 20px;
}
.probe-header h1 { margin: 0; font-size: 28px; }
.probe-header p { margin: 7px 0 0; color: var(--body-text-color-subdued); }
.probe-controls { align-items: stretch; }
.probe-controls > div { min-width: 0; }
.result-panel {
  border: 1px solid var(--border-color-primary);
  border-radius: 8px;
  padding: 20px;
  margin: 22px 0 12px;
  background: var(--block-background-fill);
}
.result-title { margin: 0 0 6px; font-size: 20px; }
.result-copy { margin: 0; color: var(--body-text-color-subdued); }
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
  margin-top: 18px;
}
.metric-card {
  border: 1px solid var(--border-color-primary);
  border-radius: 6px;
  padding: 12px;
  min-height: 92px;
}
.metric-label { color: var(--body-text-color-subdued); font-size: 12px; }
.metric-value { font-size: 25px; font-weight: 650; line-height: 1.25; margin-top: 4px; }
.metric-note { color: var(--body-text-color-subdued); font-size: 12px; margin-top: 5px; }
.probe-note { margin: 16px 0 0; color: var(--body-text-color-subdued); font-size: 13px; }
@media (max-width: 900px) {
  .metric-grid { grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); }
}
@media (max-width: 560px) {
  .gradio-container { padding: 20px 14px 36px !important; }
  .metric-grid { grid-template-columns: 1fr; }
}
"""


@dataclass(frozen=True)
class RegisteredCheckpoint:
    name: str
    checkpoint_path: Path
    comparison_model: str


@dataclass(frozen=True)
class ProbeUISettings:
    dataset_name: str
    request_name: str
    output_root: Path
    split_path: Path
    comparison_path: Path
    target_class_name: str
    prompt_template: str
    top_k: int
    baseline: RegisteredCheckpoint
    candidates: tuple[RegisteredCheckpoint, ...]


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"probe_ui.{name} must be a mapping")
    return value


def _relative_path(value: Any, *, name: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"probe_ui.{name} must be a non-empty relative path")
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"probe_ui.{name} must be a relative path")
    return path


def _registered_checkpoint(
    payload: Mapping[str, Any],
    output_root: Path,
    *,
    name: str,
) -> RegisteredCheckpoint:
    item = _mapping(payload, name=name)
    display_name = str(item.get("name", "")).strip()
    comparison_model = str(item.get("comparison_model", "")).strip()
    if not display_name or not comparison_model:
        raise ValueError(
            f"probe_ui.{name} requires non-empty name and comparison_model values"
        )
    return RegisteredCheckpoint(
        name=display_name,
        checkpoint_path=output_root
        / _relative_path(item.get("checkpoint"), name=f"{name}.checkpoint"),
        comparison_model=comparison_model,
    )


def load_probe_ui_settings(
    config_path: str,
    *,
    output_root_override: str | None = None,
) -> ProbeUISettings:
    """Load the trusted, startup-only checkpoint registry for the local UI."""
    payload = load_runtime_config(config_path)
    raw = _mapping(payload.get("probe_ui"), name="root")
    dataset_name = str(raw.get("dataset", "")).strip()
    request_name = str(raw.get("request", "")).strip()
    target_class_name = str(raw.get("target_class", "")).strip()
    if not dataset_name or not request_name or not target_class_name:
        raise ValueError("probe_ui requires dataset, request, and target_class")

    output_root = resolve_output_root(
        output_root_override,
        payload,
        dataset_name,
        request_name,
    )
    split_path = output_root / "splits" / f"{request_name}_split.json"
    comparison_path = output_root / _relative_path(
        raw.get("comparison"), name="comparison"
    )
    baseline = _registered_checkpoint(
        raw.get("baseline"), output_root, name="baseline"
    )
    candidate_payloads = raw.get("candidates")
    if not isinstance(candidate_payloads, Sequence) or isinstance(
        candidate_payloads, (str, bytes)
    ):
        raise ValueError("probe_ui.candidates must be a non-empty list")
    candidates = tuple(
        _registered_checkpoint(item, output_root, name=f"candidates[{index}]")
        for index, item in enumerate(candidate_payloads)
    )
    if not candidates:
        raise ValueError("probe_ui.candidates must not be empty")
    candidate_names = [candidate.name for candidate in candidates]
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("probe_ui candidate display names must be unique")

    model_payload = payload.get("model", {})
    prompt_template = str(
        raw.get("prompt_template", model_payload.get("prompt_template", "a photo of a {}"))
    )
    top_k = int(raw.get("top_k", 5))
    if top_k < 1:
        raise ValueError("probe_ui.top_k must be at least 1")
    return ProbeUISettings(
        dataset_name=dataset_name,
        request_name=request_name,
        output_root=output_root,
        split_path=split_path,
        comparison_path=comparison_path,
        target_class_name=target_class_name,
        prompt_template=prompt_template,
        top_k=top_k,
        baseline=baseline,
        candidates=candidates,
    )


def _validate_image(image: Any) -> Any:
    if image is None:
        raise ValueError("Choose an image before comparing checkpoints")
    if not hasattr(image, "convert") or not hasattr(image, "size"):
        raise ValueError("The uploaded file is not a decoded image")
    width, height = image.size
    if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image dimensions must contain between 1 and {MAX_IMAGE_PIXELS:,} pixels"
        )
    return image.convert("RGB")


def _top_k_frame(prediction: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Rank": row["rank"],
                "Class": row["class_name"],
                "Probability": f"{row['probability']:.4f}",
            }
            for row in prediction["top_k"]
        ]
    )


def _metric_value(row: pd.Series, key: str) -> float | None:
    value = row.get(key)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _metric_card(label: str, value: str, note: str) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-note">{html.escape(note)}</div>'
        "</div>"
    )


def _result_html(
    *,
    target_class_name: str,
    candidate_name: str,
    baseline_prediction: Mapping[str, Any],
    candidate_prediction: Mapping[str, Any],
    metrics: pd.Series,
) -> str:
    baseline_probability = float(baseline_prediction["target_probability"])
    candidate_probability = float(candidate_prediction["target_probability"])
    probability_delta = candidate_probability - baseline_probability
    target_suppressed = candidate_probability < baseline_probability
    verdict = (
        "Target suppression observed"
        if target_suppressed
        else "Target suppression was not observed"
    )
    candidate_top = candidate_prediction["top_k"][0]

    target_accuracy = _metric_value(metrics, "target_test_acc")
    sibling_accuracy = _metric_value(metrics, "sibling_test_acc")
    utility_accuracy = _metric_value(metrics, "utility_test_all")
    mia_resistance = _metric_value(metrics, "mia_resistance_confidence")
    mia_auc = _metric_value(metrics, "mia_auc_confidence")
    cards = [
        _metric_card(
            "Uploaded-image target score",
            f"{candidate_probability:.2%}",
            f"{baseline_probability:.2%} baseline; {probability_delta:+.2%} change",
        ),
        _metric_card(
            "Target test accuracy",
            f"{target_accuracy:.1%}" if target_accuracy is not None else "N/A",
            "Saved CIFAR-100 evaluation; lower is better",
        ),
        _metric_card(
            "Sibling test accuracy",
            f"{sibling_accuracy:.1%}" if sibling_accuracy is not None else "N/A",
            "Saved CIFAR-100 evaluation; higher is better",
        ),
        _metric_card(
            "Overall test accuracy",
            f"{utility_accuracy:.1%}" if utility_accuracy is not None else "N/A",
            "Saved CIFAR-100 evaluation; higher is better",
        ),
    ]
    privacy_note = (
        f"Resistance is highest at 100%; raw confidence AUC is {mia_auc:.3f}."
        if mia_auc is not None
        else "Resistance is highest at 100%."
    )
    if mia_resistance is not None:
        cards.append(
            _metric_card(
                "Confidence MIA resistance",
                f"{mia_resistance:.1%}",
                privacy_note,
            )
        )

    return (
        '<section class="result-panel">'
        f'<h2 class="result-title">{html.escape(verdict)}</h2>'
        '<p class="result-copy">'
        f'<strong>{html.escape(target_class_name)}</strong> moved from rank '
        f'{int(baseline_prediction["target_rank"])} to '
        f'{int(candidate_prediction["target_rank"])} under '
        f'<strong>{html.escape(candidate_name)}</strong>. The candidate top label is '
        f'<strong>{html.escape(str(candidate_top["class_name"]))}</strong> '
        f'({float(candidate_top["probability"]):.2%}).'
        "</p>"
        f'<div class="metric-grid">{"".join(cards)}</div>'
        '<p class="probe-note">This uploaded image is an illustrative external '
        'probe without a CIFAR-100 ground-truth label. The saved test metrics '
        'are the evidence for the experiment-level claim.</p>'
        "</section>"
    )


def _model_load_error_message(error: OSError, *, offline: bool) -> str:
    if offline:
        return (
            "The frozen CLIP base model is not available in the local cache. "
            "Run `uv run helpers/cache_model.py --dataset cifar100` once while "
            "online, then restart this interface with `--offline`."
        )
    return f"Unable to load the frozen CLIP base model: {error}"


class ProbeComparisonService:
    """Compare one immutable baseline with one selected immutable candidate."""

    def __init__(
        self,
        settings: ProbeUISettings,
        *,
        device_name: str,
        local_files_only: bool,
    ) -> None:
        self.settings = settings
        self.device_name = device_name
        self.local_files_only = local_files_only
        self._split, _spec, self.class_names = load_split_metadata(
            str(settings.split_path), settings.dataset_name
        )
        if settings.target_class_name not in self.class_names:
            raise ValueError(
                f"Target class {settings.target_class_name!r} is not in the split vocabulary"
            )
        self._validate_artifacts()
        self._comparison = pd.read_csv(settings.comparison_path)
        self._baseline: ImageCheckpointPredictor | None = None
        self._candidate: ImageCheckpointPredictor | None = None
        self._candidate_name: str | None = None
        self._lock = threading.Lock()

    def _validate_artifacts(self) -> None:
        required = [
            self.settings.split_path,
            self.settings.comparison_path,
            self.settings.baseline.checkpoint_path,
            *(candidate.checkpoint_path for candidate in self.settings.candidates),
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Probe UI requires registered artifacts that are missing: "
                + ", ".join(missing)
            )

    def _load_baseline(self) -> ImageCheckpointPredictor:
        if self._baseline is None:
            self._baseline = load_image_checkpoint_predictor(
                checkpoint_path=str(self.settings.baseline.checkpoint_path),
                dataset_name=self.settings.dataset_name,
                class_names=self.class_names,
                prompt_template=self.settings.prompt_template,
                device_name=self.device_name,
                local_files_only=self.local_files_only,
            )
        return self._baseline

    def _candidate_for_name(self, candidate_name: str) -> RegisteredCheckpoint:
        for candidate in self.settings.candidates:
            if candidate.name == candidate_name:
                return candidate
        raise ValueError("Selected checkpoint is not registered for this probe UI")

    def _load_candidate(self, candidate: RegisteredCheckpoint) -> ImageCheckpointPredictor:
        if self._candidate_name != candidate.name:
            self._candidate = None
            self._candidate_name = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if self._candidate is None:
            baseline = self._load_baseline()
            self._candidate = load_image_checkpoint_predictor(
                checkpoint_path=str(candidate.checkpoint_path),
                dataset_name=self.settings.dataset_name,
                class_names=self.class_names,
                prompt_template=self.settings.prompt_template,
                device_name=self.device_name,
                expected_model_name=baseline.model.cfg.model_name,
                local_files_only=self.local_files_only,
            )
            self._candidate_name = candidate.name
        return self._candidate

    def _metric_row(self, candidate: RegisteredCheckpoint) -> pd.Series:
        rows = self._comparison[
            self._comparison["model"] == candidate.comparison_model
        ]
        if rows.empty:
            raise ValueError(
                f"Comparison report has no row for {candidate.comparison_model!r}"
            )
        return rows.iloc[0]

    def _metric_frame(self, candidate: RegisteredCheckpoint) -> pd.DataFrame:
        row = self._metric_row(candidate)
        return pd.DataFrame(
            [
                {"Metric": label, "Value": f"{float(row[key]):.4f}"}
                for key, label in _METRIC_COLUMNS
                if key in row and pd.notna(row[key])
            ]
        )

    def compare(
        self,
        image: Any,
        candidate_name: str,
    ) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        image = _validate_image(image)
        candidate_registration = self._candidate_for_name(candidate_name)
        with self._lock:
            baseline_prediction = predict_probe_image(
                self._load_baseline(),
                image,
                target_class_name=self.settings.target_class_name,
                top_k=self.settings.top_k,
            )
            candidate_prediction = predict_probe_image(
                self._load_candidate(candidate_registration),
                image,
                target_class_name=self.settings.target_class_name,
                top_k=self.settings.top_k,
            )

        metric_row = self._metric_row(candidate_registration)
        return (
            _result_html(
                target_class_name=self.settings.target_class_name,
                candidate_name=candidate_registration.name,
                baseline_prediction=baseline_prediction,
                candidate_prediction=candidate_prediction,
                metrics=metric_row,
            ),
            _top_k_frame(baseline_prediction),
            _top_k_frame(candidate_prediction),
            self._metric_frame(candidate_registration),
        )


def create_probe_app(
    settings: ProbeUISettings,
    *,
    device_name: str,
    local_files_only: bool,
):
    try:
        import gradio as gr
    except ImportError as error:  # pragma: no cover - depends on optional UI dependency
        raise RuntimeError(
            "Gradio is required for the probe interface. Install project dependencies first."
        ) from error

    service = ProbeComparisonService(
        settings,
        device_name=device_name,
        local_files_only=local_files_only,
    )

    def compare_callback(image: Any, candidate_name: str):
        try:
            return service.compare(image, candidate_name)
        except OSError as error:
            raise gr.Error(
                _model_load_error_message(error, offline=local_files_only)
            ) from error
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            raise gr.Error(str(error)) from error

    candidate_names = [candidate.name for candidate in settings.candidates]
    with gr.Blocks(title="UN-ML Checkpoint Probe", css=_APP_CSS) as app:
        gr.HTML(
            "<header class='probe-header'>"
            "<h1>UN-ML Checkpoint Probe</h1>"
            "<p>CIFAR-100 selective unlearning: baseline versus a registered "
            "candidate checkpoint.</p>"
            "</header>"
        )
        with gr.Row(elem_classes=["probe-controls"]):
            image_input = gr.Image(
                label="Upload or paste an image",
                type="pil",
                image_mode="RGB",
                sources=["upload", "clipboard"],
                height=360,
                scale=3,
            )
            with gr.Column(scale=2, min_width=320):
                candidate_input = gr.Dropdown(
                    choices=candidate_names,
                    value=candidate_names[0],
                    label="Unlearned checkpoint",
                )
                gr.Markdown(
                    f"**Experiment:** `{settings.dataset_name}` / "
                    f"`{settings.request_name}`\n\n"
                    f"**Target class:** `{settings.target_class_name}`"
                )
                compare_button = gr.Button(
                    "Compare checkpoints", variant="primary", size="lg"
                )
        result = gr.HTML()
        with gr.Accordion("Prediction details", open=False):
            with gr.Row():
                baseline_table = gr.Dataframe(
                    headers=["Rank", "Class", "Probability"],
                    label=settings.baseline.name,
                    interactive=False,
                    max_height=250,
                )
                candidate_table = gr.Dataframe(
                    headers=["Rank", "Class", "Probability"],
                    label="Selected unlearned checkpoint",
                    interactive=False,
                    max_height=250,
                )
        with gr.Accordion("Saved aggregate evaluation metrics", open=False):
            metrics_table = gr.Dataframe(
                headers=["Metric", "Value"],
                label="Candidate evaluation record",
                interactive=False,
                max_height=280,
            )
        compare_button.click(
            compare_callback,
            inputs=[image_input, candidate_input],
            outputs=[result, baseline_table, candidate_table, metrics_table],
        )
    return app
