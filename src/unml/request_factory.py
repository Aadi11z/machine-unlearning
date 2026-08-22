from __future__ import annotations

from dataclasses import dataclass

from .config import normalize_dataset_name
from .data import (
    DatasetSpec,
    SplitConfig,
    download_and_prepare_splits,
    get_dataset_spec,
    validate_hierarchy_request,
)


@dataclass(frozen=True)
class SelectiveRequest:
    """A single-class deletion request derived from CIFAR hierarchy metadata."""

    class_id: int
    class_name: str
    superclass: str
    sibling_classes: tuple[int, ...]
    request_name: str


def _normalize_class_token(value: object) -> str:
    return str(value).strip().lower().replace("_", " ")


def _sanitize_request_name(class_name: str) -> str:
    sanitized = "_".join(class_name.strip().lower().split())
    if not sanitized:
        raise ValueError("Class name produced an empty request name")
    return f"{sanitized}_selective"


def resolve_selective_request(
    dataset_name: str, target: int | str
) -> SelectiveRequest:
    spec = get_dataset_spec(dataset_name)
    if normalize_dataset_name(dataset_name) != "cifar100":
        raise ValueError(
            "Selective requests currently require cifar100 hierarchy metadata"
        )
    class_id, class_name = _resolve_class(spec, target)
    superclass = _superclass_for_class(spec, class_id)
    siblings = tuple(
        sorted(set(spec.superclasses[superclass]) - {class_id})
    )
    return SelectiveRequest(
        class_id=class_id,
        class_name=class_name,
        superclass=superclass,
        sibling_classes=siblings,
        request_name=_sanitize_request_name(class_name),
    )


def _resolve_class(
    spec: DatasetSpec, target: int | str
) -> tuple[int, str]:
    class_names = list(spec.class_names)
    if isinstance(target, bool):
        raise ValueError(f"Invalid class target: {target!r}")
    if isinstance(target, int):
        class_id = int(target)
        if class_id < 0 or class_id >= len(class_names):
            raise ValueError(
                f"Class id {class_id} outside [0, {len(class_names) - 1}]"
            )
        return class_id, class_names[class_id]

    normalized = _normalize_class_token(target)
    if not normalized:
        raise ValueError("Class name must be non-empty")
    if normalized.isdigit():
        return _resolve_class(spec, int(normalized))
    lookup = {
        _normalize_class_token(name): index
        for index, name in enumerate(class_names)
    }
    if normalized not in lookup:
        raise ValueError(
            f"Unknown class name={target!r}; choose from {class_names}"
        )
    class_id = lookup[normalized]
    return class_id, class_names[class_id]


def _superclass_for_class(spec: DatasetSpec, class_id: int) -> str:
    for superclass, members in spec.superclasses.items():
        if class_id in members:
            return superclass
    raise ValueError(f"Class id {class_id} belongs to no known superclass")


def selective_split_config(
    request: SelectiveRequest,
    *,
    forget_fraction: float,
    retain_val_fraction: float,
    seed: int,
) -> SplitConfig:
    return SplitConfig(
        forget_classes=[request.class_id],
        forget_fraction=forget_fraction,
        retain_val_fraction=retain_val_fraction,
        seed=seed,
        request_name=request.request_name,
        request_type="selective_class",
        superclass=request.superclass,
        sibling_classes=request.sibling_classes,
    )


def validate_selective_request(
    spec: DatasetSpec, cfg: SplitConfig
) -> None:
    validate_hierarchy_request(spec, cfg)


def build_selective_split(
    data_dir: str,
    split_path: str,
    request: SelectiveRequest,
    *,
    forget_fraction: float,
    retain_val_fraction: float,
    seed: int,
    dataset_name: str = "cifar100",
) -> dict:
    """Create and persist the split for an on-demand single-class request."""
    return download_and_prepare_splits(
        data_dir=data_dir,
        split_path=split_path,
        forget_classes=[request.class_id],
        forget_fraction=forget_fraction,
        retain_val_fraction=retain_val_fraction,
        seed=seed,
        dataset_name=dataset_name,
        request_name=request.request_name,
        request_type="selective_class",
        superclass=request.superclass,
        sibling_classes=list(request.sibling_classes),
    )


def list_superclass_groups(
    dataset_name: str = "cifar100",
) -> list[dict[str, object]]:
    """Return superclass groups in UI-friendly order."""
    spec = get_dataset_spec(dataset_name)
    groups = []
    for superclass, member_ids in spec.superclasses.items():
        groups.append(
            {
                "superclass": superclass,
                "class_ids": list(member_ids),
                "class_names": [spec.class_names[class_id] for class_id in member_ids],
            }
        )
    return groups
