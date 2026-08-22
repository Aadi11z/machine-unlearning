from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any, Callable, Mapping, Sequence

import torch

from unml.model import (
    apply_adapter_state,
    read_checkpoint_payload,
    validate_adapter_payload,
)
from unml.probe import ImageCheckpointPredictor, predict_probe_image


PredictionLoader = Callable[..., ImageCheckpointPredictor]
_PREDICTION_CACHE_CAP = 64


def _image_fingerprint(image: Any) -> str:
    if not hasattr(image, "convert"):
        raise ValueError("Probe image must be a decoded image object")
    rgb = image.convert("RGB")
    digest = hashlib.sha1()
    digest.update(str(rgb.size).encode("utf-8"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


class ModelRegistry:
    """Hold one loaded checkpoint and swap adapter deltas without rebuilds.

    The frozen CLIP backbone, image processor, and class text features are
    created once. Activating another checkpoint copies only its trainable
    adapter weights onto the live model, so switching is effectively free and
    many checkpoints can be served from a single resident backbone.
    """

    def __init__(
        self,
        *,
        baseline_checkpoint_path: str,
        baseline_name: str,
        dataset_name: str,
        class_names: Sequence[str],
        prompt_template: str,
        device_name: str = "auto",
        local_files_only: bool = False,
        predictor_loader: PredictionLoader | None = None,
    ) -> None:
        if predictor_loader is None:
            from unml.probe import load_image_checkpoint_predictor

            predictor_loader = load_image_checkpoint_predictor
        self._predictor: ImageCheckpointPredictor = predictor_loader(
            checkpoint_path=baseline_checkpoint_path,
            dataset_name=dataset_name,
            class_names=list(class_names),
            prompt_template=prompt_template,
            device_name=device_name,
            local_files_only=local_files_only,
        )
        model = self._predictor.model
        if not model.can_cache_text_features_during_training():
            raise ValueError(
                "ModelRegistry requires a frozen text tower; the loaded "
                "checkpoint trains text-side adapters so per-checkpoint text "
                "features cannot be shared"
            )
        payload = read_checkpoint_payload(baseline_checkpoint_path)
        state = payload.get("adapter_state_dict")
        if not isinstance(state, Mapping) or not state:
            raise ValueError(
                f"Baseline checkpoint {baseline_checkpoint_path!r} contains "
                "no adapter weights"
            )
        self._baseline_state = dict(state)
        self.baseline_name = baseline_name
        self._active_name = baseline_name
        self._lock = threading.RLock()
        self._prediction_cache: OrderedDict[tuple, dict] = OrderedDict()

    @property
    def class_names(self) -> Sequence[str]:
        return self._predictor.class_names

    @property
    def active_name(self) -> str:
        return self._active_name

    def activate(self, name: str, checkpoint_path: str) -> None:
        with self._lock:
            payload = read_checkpoint_payload(checkpoint_path)
            validate_adapter_payload(
                payload,
                self._predictor.model.cfg,
                source=name,
                expected_model_name=self._predictor.model.cfg.model_name,
            )
            apply_adapter_state(self._predictor.model, payload["adapter_state_dict"])
            self._active_name = name

    def reset_to_baseline(self) -> None:
        with self._lock:
            apply_adapter_state(self._predictor.model, self._baseline_state)
            self._active_name = self.baseline_name

    def predict(
        self,
        image: Any,
        *,
        target_class_name: str,
        top_k: int,
    ) -> dict:
        with self._lock:
            return predict_probe_image(
                self._predictor,
                image,
                target_class_name=target_class_name,
                top_k=top_k,
            )

    def _cached_predict(
        self,
        cache_scope: str,
        image: Any,
        *,
        target_class_name: str,
        top_k: int,
    ) -> dict:
        fingerprint = (cache_scope, _image_fingerprint(image), target_class_name, top_k)
        cached = self._prediction_cache.get(fingerprint)
        if cached is not None:
            self._prediction_cache.move_to_end(fingerprint)
            return cached
        prediction = self.predict(image, target_class_name=target_class_name, top_k=top_k)
        self._prediction_cache[fingerprint] = prediction
        while len(self._prediction_cache) > _PREDICTION_CACHE_CAP:
            self._prediction_cache.popitem(last=False)
        return prediction

    def compare(
        self,
        image: Any,
        *,
        candidate_name: str,
        candidate_checkpoint_path: str,
        target_class_name: str,
        top_k: int,
    ) -> dict:
        """Predict under baseline weights, then under one candidate's weights."""
        with self._lock:
            self.reset_to_baseline()
            try:
                baseline_prediction = self._cached_predict(
                    f"baseline:{target_class_name}:{top_k}",
                    image,
                    target_class_name=target_class_name,
                    top_k=top_k,
                )
                self.activate(candidate_name, candidate_checkpoint_path)
                candidate_prediction = self.predict(
                    image, target_class_name=target_class_name, top_k=top_k
                )
            finally:
                self.reset_to_baseline()
        return {
            "baseline_prediction": baseline_prediction,
            "candidate_prediction": candidate_prediction,
            "candidate_name": candidate_name,
        }
