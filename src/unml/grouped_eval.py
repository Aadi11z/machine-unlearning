from __future__ import annotations

from typing import Any, Dict


def _group_classes(
    split: Dict[str, Any],
    num_classes: int,
) -> Dict[str, list[int]]:
    target = sorted(
        set(split.get("target_classes", split.get("forget_classes", [])))
    )
    sibling = sorted(set(split.get("sibling_classes", [])))
    unrelated = sorted(set(split.get("unrelated_classes", [])))
    if not unrelated:
        unrelated = sorted(set(range(num_classes)) - set(target) - set(sibling))
    if not target:
        raise ValueError("Evaluation requires at least one target class")

    named_groups = {
        "target": set(target),
        "sibling": set(sibling),
        "unrelated": set(unrelated),
    }
    for group_name, class_ids in named_groups.items():
        invalid = sorted(
            class_id
            for class_id in class_ids
            if class_id < 0 or class_id >= num_classes
        )
        if invalid:
            raise ValueError(
                f"{group_name} classes outside [0, {num_classes - 1}]: "
                f"{invalid}"
            )

    for left, right in (
        ("target", "sibling"),
        ("target", "unrelated"),
        ("sibling", "unrelated"),
    ):
        overlap = sorted(named_groups[left] & named_groups[right])
        if overlap:
            raise ValueError(
                f"Evaluation class groups {left}/{right} overlap: {overlap}"
            )

    missing = sorted(set(range(num_classes)) - set().union(*named_groups.values()))
    if missing:
        raise ValueError(f"Evaluation class groups do not cover classes: {missing}")
    return {
        "target": target,
        "sibling": sibling,
        "unrelated": unrelated,
        "retain": sorted(set(sibling) | set(unrelated)),
        "all": list(range(num_classes)),
    }
