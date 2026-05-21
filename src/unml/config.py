from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


def load_runtime_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError("Top-level YAML config must be a mapping")
    return payload


def nested_get(payload: dict[str, Any], path: Sequence[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def resolve_value(cli_value: Any, payload: dict[str, Any], path: Sequence[str], fallback: Any) -> Any:
    if cli_value is not None:
        return cli_value
    yaml_value = nested_get(payload, path)
    if yaml_value is not None:
        return yaml_value
    return fallback


def resolve_str_list(cli_value: str | None, payload: dict[str, Any], path: Sequence[str], fallback: Iterable[str]) -> list[str]:
    if cli_value is not None:
        return [item.strip() for item in cli_value.split(",") if item.strip()]

    yaml_value = nested_get(payload, path)
    if yaml_value is not None:
        if isinstance(yaml_value, str):
            return [item.strip() for item in yaml_value.split(",") if item.strip()]
        if isinstance(yaml_value, list):
            return [str(item).strip() for item in yaml_value if str(item).strip()]
        raise ValueError(f"Expected list or comma-separated string at {'.'.join(path)}")

    return [item.strip() for item in fallback if str(item).strip()]


def parse_forget_classes(raw: str | Sequence[int]) -> list[int]:
    if isinstance(raw, str):
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    return [int(item) for item in raw]

