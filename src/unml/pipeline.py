from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from .utils import git_commit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprints(paths: Sequence[str | Path]) -> list[dict[str, object]]:
    fingerprints = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Required pipeline artifact not found: {path}")
        fingerprints.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return fingerprints


def stage_receipt_path(output_root: str | Path, stage: str) -> Path:
    safe_stage = stage.replace(":", "__").replace("/", "_")
    return Path(output_root) / ".pipeline" / "stages" / f"{safe_stage}.json"


def stage_contract(
    *,
    stage: str,
    command: Sequence[str],
    input_paths: Sequence[str | Path],
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    return {
        "stage": stage,
        "git_commit": git_commit(repo_root),
        "command": [str(value) for value in command],
        "inputs": _fingerprints(input_paths),
    }


def validate_stage_receipt(
    receipt_path: str | Path,
    *,
    contract: dict[str, object],
    output_paths: Sequence[str | Path],
) -> tuple[bool, str]:
    path = Path(receipt_path)
    if not path.is_file():
        return False, "receipt missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return False, f"receipt unreadable: {error}"
    if payload.get("contract") != contract:
        return False, "stage contract changed"
    try:
        current_outputs = _fingerprints(output_paths)
    except FileNotFoundError as error:
        return False, str(error)
    if payload.get("outputs") != current_outputs:
        return False, "stage outputs changed"
    return True, "validated receipt and artifacts"


def write_stage_receipt(
    receipt_path: str | Path,
    *,
    contract: dict[str, object],
    output_paths: Sequence[str | Path],
) -> Path:
    path = Path(receipt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "outputs": _fingerprints(output_paths),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def run_pipeline_stage(
    *,
    stage: str,
    command: Sequence[str],
    env: dict[str, str],
    output_root: str | Path,
    input_paths: Sequence[str | Path],
    output_paths: Sequence[str | Path],
    resume: bool,
    force: bool = False,
    repo_root: str | Path | None = None,
    runner: Callable[..., object] = subprocess.run,
) -> str:
    contract = stage_contract(
        stage=stage,
        command=command,
        input_paths=input_paths,
        repo_root=repo_root,
    )
    receipt = stage_receipt_path(output_root, stage)
    if resume and not force:
        valid, reason = validate_stage_receipt(
            receipt,
            contract=contract,
            output_paths=output_paths,
        )
        if valid:
            print(f"[skip] {stage}: {reason}", flush=True)
            return "skipped"
        print(f"[resume] {stage}: {reason}; running stage", flush=True)

    print("[cmd]", " ".join(command), flush=True)
    runner(list(command), check=True, env=env)
    write_stage_receipt(
        receipt,
        contract=contract,
        output_paths=output_paths,
    )
    print(f"[receipt] {stage}: {receipt}", flush=True)
    return "completed"
