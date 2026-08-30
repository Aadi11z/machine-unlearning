#!/usr/bin/env python3
"""Create a target-specific retain/forget split for a canonical oracle run."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from _bootstrap import REPO_ROOT, configure_runtime
except ModuleNotFoundError:
    from scripts._bootstrap import REPO_ROOT, configure_runtime

configure_runtime()

from unml.config import (  # noqa: E402
    load_runtime_config,
    parse_forget_classes,
    parse_optional_classes,
    resolve_request_config,
)
from unml.data import download_and_prepare_splits  # noqa: E402
from unml.data import summarize_splits  # noqa: E402


def resolve_oracle_request(
    *,
    config_path: str | Path,
    request_name: str,
    forget_classes_override: str | None = None,
) -> dict[str, object]:
    runtime_config = load_runtime_config(str(config_path))
    request = resolve_request_config(request_name, runtime_config, "cifar100")
    if request is None:
        raise ValueError(f"Unknown CIFAR-100 oracle request={request_name!r}")

    required = ("request_type", "superclass", "target_classes")
    missing = [key for key in required if request.get(key) is None]
    if missing:
        raise ValueError(
            f"Oracle request={request_name!r} is missing fields: {missing}"
        )

    target_classes = parse_forget_classes(request["target_classes"])
    if forget_classes_override is not None:
        override = parse_forget_classes(forget_classes_override)
        if sorted(set(override)) != sorted(set(target_classes)):
            raise ValueError(
                f"--forget-classes {override} does not match configured "
                f"target_classes {target_classes} for request={request_name!r}"
            )

    return {
        "request_name": request_name,
        "request_type": str(request["request_type"]),
        "superclass": str(request["superclass"]),
        "target_classes": target_classes,
        "sibling_classes": parse_optional_classes(request.get("sibling_classes")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "parameters.yaml",
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument(
        "--forget-classes",
        default=None,
        help=(
            "Optional compatibility check. When supplied, the class ids must "
            "match the configured target_classes for --request."
        ),
    )
    parser.add_argument("--forget-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    request = resolve_oracle_request(
        config_path=args.config,
        request_name=args.request,
        forget_classes_override=args.forget_classes,
    )
    download_and_prepare_splits(
        data_dir=str(args.data_dir),
        split_path=str(args.output_path),
        forget_classes=request["target_classes"],
        forget_fraction=args.forget_fraction,
        retain_val_fraction=0.1,
        seed=args.seed,
        dataset_name="cifar100",
        request_name=str(request["request_name"]),
        request_type=str(request["request_type"]),
        superclass=str(request["superclass"]),
        sibling_classes=request["sibling_classes"],
    )
    print(f"[oracle-split] path={args.output_path}")
    print(f"[oracle-split] summary={summarize_splits(str(args.output_path))}")


if __name__ == "__main__":
    main()
