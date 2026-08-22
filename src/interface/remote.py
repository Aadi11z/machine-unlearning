"""Remote-execution primitives shared by the platform and the Modal worker."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path

SIGNATURE_HEADER = "X-UNML-Signature"


def sign_payload(secret: str, body: bytes) -> str:
    return hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()


def verify_signature(secret: str, body: bytes, signature: str | None) -> bool:
    if not signature:
        return False
    expected = sign_payload(secret, body)
    return hmac.compare_digest(expected, signature.strip().lower())


def encode_checkpoint(checkpoint_bytes: bytes) -> str:
    return base64.b64encode(checkpoint_bytes).decode("ascii")


def decode_checkpoint(encoded: str) -> bytes:
    return base64.b64decode(encoded.encode("ascii"))


def parse_job_response(payload: bytes) -> dict:
    parsed = json.loads(payload)
    required = {"checkpoint_b64", "class_id", "class_name", "request_name",
                "superclass", "sibling_classes", "method", "steps"}
    missing = sorted(required - set(parsed))
    if missing:
        raise ValueError(f"Worker response missing fields: {missing}")
    if not parsed["checkpoint_b64"]:
        raise ValueError("Worker response has an empty checkpoint")
    return parsed


def write_job_artifacts(
    *,
    output_root: Path,
    class_id: int,
    class_name: str,
    request_name: str,
    superclass: str,
    sibling_classes: list[int],
    method: str,
    steps: int,
    checkpoint_bytes: bytes,
    metrics: dict,
    wall_time_s: float,
) -> Path:
    """Persist a remote result in the layout ArtifactCatalog already scans.

    Remote deltas arrive as safetensors bytes and are validated without any
    pickle deserialization; only the trusted local baseline is read via
    torch.load.
    """
    import torch
    from safetensors.torch import load as st_load

    baseline_payload = torch.load(
        _baseline_for(output_root), map_location="cpu", weights_only=False
    )
    reference_cfg = _cfg_from(baseline_payload["model_config"])

    from unml.model import (
        LightweightVLM,
        apply_adapter_state,
        read_safetensors_metadata,
        validate_adapter_payload,
    )

    metadata = read_safetensors_metadata(checkpoint_bytes)
    scalar_keys = set(json.loads(metadata.get("scalar_keys", "[]")))
    remote_tensors = {
        key: value.reshape(()) if key in scalar_keys else value
        for key, value in st_load(checkpoint_bytes).items()
    }
    if not remote_tensors:
        raise ValueError("Remote worker returned an empty safetensors payload")
    stored_cfg = json.loads(metadata["model_config"])
    validate_adapter_payload(
        {"model_config": stored_cfg, "adapter_state_dict": remote_tensors},
        reference_cfg,
        source=f"remote:{request_name}_{method}",
    )
    # Validate the complete tensor contract against a fresh baseline-shaped
    # model before persisting anything.  This keeps malformed remote deltas
    # from becoming apparently successful jobs that fail only when probed.
    validation_model = LightweightVLM.from_config(reference_cfg)
    apply_adapter_state(validation_model, remote_tensors)

    job_dir = (
        output_root
        / "cifar100"
        / "jobs"
        / f"{request_name}_{method}_{steps}"
    )
    checkpoint_path = job_dir / "checkpoints" / f"unlearn_{method}.safetensors"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(checkpoint_bytes)

    job_result = {
        "candidate_id": f"{request_name}_{method}_{steps}",
        "class_id": class_id,
        "class_name": class_name,
        "request_name": request_name,
        "superclass": superclass,
        "sibling_classes": sibling_classes,
        "method": method,
        "steps": steps,
        "wall_time_s": wall_time_s,
        "result": {"checkpoint": str(checkpoint_path), "metrics": metrics},
    }
    (job_dir / "job_result.json").write_text(
        json.dumps(job_result, indent=2), encoding="utf-8"
    )
    return checkpoint_path


def _baseline_for(output_root: Path) -> Path:
    matches = sorted((output_root / "cifar100").glob("*/baseline_*/checkpoints/finetuned_best.pt"))
    if not matches:
        raise FileNotFoundError(
            "No baseline checkpoint under output root; cannot validate remote delta"
        )
    return matches[0]


def _cfg_from(stored_cfg: dict):
    from unml.model import ModelConfig

    return ModelConfig(**stored_cfg)
