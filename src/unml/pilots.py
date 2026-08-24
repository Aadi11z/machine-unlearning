from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


PILOT_REPORT_SCHEMA = "unml-lora-pilot-report-v1"
DEFAULT_SCHEMA_SCREEN_STEPS = 250
DEFAULT_VALIDATION_TOLERANCE = 0.005


@dataclass(frozen=True)
class PilotSpec:
    name: str
    stage: str
    seed: int
    lora_rank: int
    lora_alpha: float
    lora_targets: tuple[str, ...]
    lora_dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    random_crop: bool = False
    warmup_fraction: float = 0.1
    max_train_steps: int = DEFAULT_SCHEMA_SCREEN_STEPS

    def to_config(self) -> dict[str, Any]:
        config = asdict(self)
        config["lora_targets"] = list(self.lora_targets)
        return config


def schema_screen_specs(*, seed: int = 42, steps: int = DEFAULT_SCHEMA_SCREEN_STEPS) -> list[PilotSpec]:
    specs: list[PilotSpec] = []
    for rank in (4, 8, 16):
        specs.append(
            PilotSpec(
                name=f"qv_r{rank}",
                stage="schema_screen",
                seed=seed,
                lora_rank=rank,
                lora_alpha=float(rank),
                lora_targets=("q_proj", "v_proj"),
                max_train_steps=steps,
            )
        )
    specs.append(
        PilotSpec(
            name="qkv_r8",
            stage="schema_screen",
            seed=seed,
            lora_rank=8,
            lora_alpha=8.0,
            lora_targets=("q_proj", "k_proj", "v_proj"),
            max_train_steps=steps,
        )
    )
    return specs


def optimization_screen_specs(
    leading: Iterable[PilotSpec],
    *,
    seed: int = 42,
    steps: int = DEFAULT_SCHEMA_SCREEN_STEPS,
) -> list[PilotSpec]:
    specs: list[PilotSpec] = []
    for schema in leading:
        for lr in (3e-4, 1e-3, 3e-3):
            for weight_decay in (1e-4, 1e-2, 5e-2):
                specs.append(
                    PilotSpec(
                        name=f"{schema.name}_lr{lr:g}_wd{weight_decay:g}",
                        stage="optimization_screen",
                        seed=seed,
                        lora_rank=schema.lora_rank,
                        lora_alpha=schema.lora_alpha,
                        lora_targets=schema.lora_targets,
                        lr=lr,
                        weight_decay=weight_decay,
                        max_train_steps=steps,
                    )
                )
    return specs


def confirmation_specs(
    leading: Iterable[PilotSpec], *, seeds: tuple[int, ...] = (42, 123, 456)
) -> list[PilotSpec]:
    specs: list[PilotSpec] = []
    for schema in leading:
        for seed in seeds:
            specs.append(
                PilotSpec(
                    name=f"{schema.name}_s{seed}",
                    stage="confirmation",
                    seed=seed,
                    lora_rank=schema.lora_rank,
                    lora_alpha=schema.lora_alpha,
                    lora_targets=schema.lora_targets,
                    lora_dropout=schema.lora_dropout,
                    lr=schema.lr,
                    weight_decay=schema.weight_decay,
                    random_crop=schema.random_crop,
                    warmup_fraction=schema.warmup_fraction,
                    max_train_steps=schema.max_train_steps,
                )
            )
    return specs


def _result_score(result: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(result.get("retain_val_acc", float("-inf"))),
        float(result.get("retain_val_macro_accuracy", float("-inf"))),
        -float(result.get("adapter_size_bytes", float("inf"))),
    )


def rank_pilot_results(
    results: Iterable[Mapping[str, Any]],
    *,
    smaller_adapter_tolerance: float = DEFAULT_VALIDATION_TOLERANCE,
) -> list[dict[str, Any]]:
    """Rank validation-only results and prefer a smaller near-tie adapter."""
    ranked = [dict(result) for result in results]
    for result in ranked:
        metrics = result.get("metrics", result)
        if "test_all_acc" in metrics or "test_retain_acc" in metrics:
            raise ValueError("Pilot results must not contain test-set metrics")
        result["retain_val_acc"] = float(metrics["retain_val_acc"])
        result["retain_val_macro_accuracy"] = float(
            metrics.get("retain_val_macro_accuracy", metrics["retain_val_acc"])
        )
        result["adapter_size_bytes"] = int(result.get("adapter_size_bytes", 0))
    ranked.sort(key=_result_score, reverse=True)
    if ranked:
        best_score = ranked[0]["retain_val_acc"]
        near_ties = [
            item
            for item in ranked
            if best_score - item["retain_val_acc"] <= smaller_adapter_tolerance
        ]
        winner = min(
            near_ties,
            key=lambda item: (item["adapter_size_bytes"], -item["retain_val_acc"]),
        )
        ranked.remove(winner)
        ranked.insert(0, winner)
    for index, result in enumerate(ranked, 1):
        result["rank"] = index
    return ranked


def build_pilot_report(
    *,
    split_path: str,
    split_digest: str,
    prompt_contract: Mapping[str, Any],
    results: Iterable[Mapping[str, Any]],
    winner: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ranked = rank_pilot_results(results)
    return {
        "schema": PILOT_REPORT_SCHEMA,
        "dataset": "cifar100",
        "test_metrics_used": False,
        "split": {"path": split_path, "sha256": split_digest},
        "prompt_contract": dict(prompt_contract),
        "results": ranked,
        "winner": dict(winner or (ranked[0] if ranked else {})),
    }


def write_pilot_report(path: str | Path, report: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
