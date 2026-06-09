from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


DEFAULT_FORGET_CLASSES = {
    "cifar10": "3,5",
    "cifar100": "54,62,70,82,92",
}


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


def parse_optional_classes(raw: str | Sequence[int] | None) -> list[int]:
    if raw is None:
        return []
    return parse_forget_classes(raw)


def normalize_dataset_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "cifar10": "cifar10",
        "cifar100": "cifar100",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported dataset={value!r}. Choose from {sorted(aliases)}"
        )
    return aliases[normalized]


def resolve_dataset_and_split_path(
    cli_dataset: str | None,
    cli_split_path: str | None,
    payload: dict[str, Any],
    cli_request: str | None = None,
) -> tuple[str, str]:
    configured_dataset = normalize_dataset_name(
        str(resolve_value(None, payload, ("data", "dataset"), "cifar10"))
    )
    dataset_name = normalize_dataset_name(
        str(cli_dataset) if cli_dataset is not None else configured_dataset
    )

    if cli_split_path is not None:
        return dataset_name, cli_split_path

    profile_split = nested_get(
        payload, ("data", "profiles", dataset_name, "split_path")
    )
    if profile_split is not None:
        rendered_split = str(profile_split).format(
            dataset=dataset_name,
            output_root=resolve_output_root(
                None, payload, dataset_name, cli_request
            ),
        )
        return dataset_name, rendered_split

    configured_split = nested_get(payload, ("data", "split_path"))
    if configured_split is not None and dataset_name == configured_dataset:
        return dataset_name, str(configured_split)

    output_root = resolve_output_root(None, payload, dataset_name, cli_request)
    request_name = resolve_request_name(cli_request, payload, dataset_name)
    split_stem = request_name or dataset_name
    return dataset_name, str(
        output_root / "splits" / f"{split_stem}_split.json"
    )


def resolve_forget_classes(
    cli_value: str | None,
    payload: dict[str, Any],
    dataset_name: str,
    cli_request: str | None = None,
) -> str:
    dataset_name = normalize_dataset_name(dataset_name)
    if cli_value is not None:
        return cli_value

    request = resolve_request_config(cli_request, payload, dataset_name)
    if request is not None and request.get("target_classes") is not None:
        return str(request["target_classes"])

    profile_classes = nested_get(
        payload, ("data", "profiles", dataset_name, "forget_classes")
    )
    if profile_classes is not None:
        return str(profile_classes)

    configured_dataset = normalize_dataset_name(
        str(resolve_value(None, payload, ("data", "dataset"), "cifar10"))
    )
    configured_classes = nested_get(payload, ("data", "forget_classes"))
    if configured_classes is not None and dataset_name == configured_dataset:
        return str(configured_classes)

    return DEFAULT_FORGET_CLASSES[dataset_name]


def resolve_request_name(
    cli_request: str | None,
    payload: dict[str, Any],
    dataset_name: str,
) -> str | None:
    if cli_request is not None:
        request_name = cli_request.strip()
        return request_name or None

    dataset_name = normalize_dataset_name(dataset_name)
    configured = nested_get(payload, ("data", "profiles", dataset_name, "request"))
    if configured is None:
        return None
    request_name = str(configured).strip()
    return request_name or None


def resolve_request_config(
    cli_request: str | None,
    payload: dict[str, Any],
    dataset_name: str,
) -> dict[str, Any] | None:
    dataset_name = normalize_dataset_name(dataset_name)
    request_name = resolve_request_name(cli_request, payload, dataset_name)
    if request_name is None:
        return None

    request = nested_get(
        payload,
        ("data", "profiles", dataset_name, "requests", request_name),
    )
    if not isinstance(request, dict):
        raise ValueError(
            f"Unknown request={request_name!r} for dataset={dataset_name!r}"
        )
    return {"name": request_name, **request}


def resolve_dataset_value(
    cli_value: Any,
    payload: dict[str, Any],
    dataset_name: str,
    key: str,
    fallback: Any,
) -> Any:
    if cli_value is not None:
        return cli_value

    dataset_name = normalize_dataset_name(dataset_name)
    profile_value = nested_get(payload, ("data", "profiles", dataset_name, key))
    if profile_value is not None:
        return profile_value

    shared_value = nested_get(payload, ("data", key))
    if shared_value is not None:
        return shared_value

    return fallback


def resolve_model_value(
    cli_value: Any,
    payload: dict[str, Any],
    dataset_name: str,
    key: str,
    fallback: Any,
) -> Any:
    """Resolve a model setting from CLI, dataset profile, shared config, fallback."""
    if cli_value is not None:
        return cli_value

    dataset_name = normalize_dataset_name(dataset_name)
    profile_value = nested_get(
        payload, ("model", "profiles", dataset_name, key)
    )
    if profile_value is not None:
        return profile_value

    shared_value = nested_get(payload, ("model", key))
    if shared_value is not None:
        return shared_value

    return fallback


def resolve_section_value(
    cli_value: Any,
    payload: dict[str, Any],
    dataset_name: str,
    section: str,
    key: str,
    fallback: Any,
) -> Any:
    """Resolve a stage setting with an optional dataset-specific profile."""
    if cli_value is not None:
        return cli_value

    dataset_name = normalize_dataset_name(dataset_name)
    profile_value = nested_get(
        payload, (section, "profiles", dataset_name, key)
    )
    if profile_value is not None:
        return profile_value

    shared_value = nested_get(payload, (section, key))
    if shared_value is not None:
        return shared_value

    return fallback


def resolve_output_root(
    cli_output_root: str | None,
    payload: dict[str, Any],
    dataset_name: str,
    cli_request: str | None = None,
) -> Path:
    dataset_name = normalize_dataset_name(dataset_name)
    if cli_output_root is not None:
        return Path(cli_output_root)

    request_name = resolve_request_name(cli_request, payload, dataset_name)
    environment_root = os.environ.get("UNML_OUTPUTS")
    if environment_root:
        output_root = Path(environment_root) / dataset_name
        return output_root / request_name if request_name else output_root

    raw_root = nested_get(payload, ("outputs", "root"))
    if raw_root is None:
        raw_root = "outputs/{dataset}"

    rendered = str(raw_root).format(dataset=dataset_name)
    output_root = Path(rendered)
    return output_root / request_name if request_name else output_root


def resolve_data_dir(
    cli_data_dir: str | None,
    payload: dict[str, Any],
) -> Path:
    if cli_data_dir is not None:
        return Path(cli_data_dir)

    environment_data = os.environ.get("UNML_DATA")
    if environment_data:
        return Path(environment_data)

    return Path(str(resolve_value(None, payload, ("data", "data_dir"), "data")))


def resolve_stage_output_dir(
    cli_output_dir: str | None,
    payload: dict[str, Any],
    dataset_name: str,
    stage: str,
    cli_request: str | None = None,
) -> Path:
    if cli_output_dir is not None:
        return Path(cli_output_dir)

    stage_names = {
        "training": "finetune",
        "unlearning": "unlearning",
        "attack": "comparison",
    }
    if stage not in stage_names:
        raise ValueError(f"Unsupported output stage={stage!r}")

    if os.environ.get("UNML_OUTPUTS") or nested_get(payload, ("outputs", "root")):
        return (
            resolve_output_root(None, payload, dataset_name, cli_request)
            / stage_names[stage]
        )

    configured_stage_dir = nested_get(payload, (stage, "output_dir"))
    if configured_stage_dir is not None:
        return Path(str(configured_stage_dir).format(dataset=dataset_name))

    return (
        resolve_output_root(None, payload, dataset_name, cli_request)
        / stage_names[stage]
    )
