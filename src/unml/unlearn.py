from __future__ import annotations

import copy
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import trange
from transformers import CLIPImageProcessor, CLIPTokenizer

from .data import (
    build_loaders,
    build_text_inputs,
    cycle_loader,
    load_split_metadata,
    validate_checkpoint_dataset,
)
from .evaluate import build_class_text_features, evaluate_classification
from .disentangle import (
    SemanticBases,
    build_semantic_bases,
    collect_class_image_prototypes,
    h_tgsd_objective,
)
from .model import load_checkpoint, precision_context, save_checkpoint
from helpers.tracker import log_unlearn_run
from .utils import (
    collect_run_provenance,
    format_metrics,
    get_device,
    move_to_device,
    save_json,
    set_seed,
    tensor_to_float,
)


UNLEARNING_METHODS = {
    "retain_only",
    "ga_kl",
    "counterfactual_rebind",
    "entropy_rebind",
    "h_tgsd",
    "h_tgsd_no_sibling_preservation",
}
H_TGSD_METHODS = {"h_tgsd", "h_tgsd_no_sibling_preservation"}


@dataclass
class UnlearnConfig:
    data_dir: str
    split_path: str
    finetuned_checkpoint: str
    output_dir: str
    method: str
    dataset_name: str = "cifar10"
    model_name: str = "openai/clip-vit-base-patch32"
    prompt_template: str = "a photo of a {}"
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2
    non_blocking: bool = False
    steps: int = 500
    lr: float = 5e-4
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "auto"
    kl_temperature: float = 1.5
    kl_weight: float = 1.0
    ga_weight: float = 1.0
    cf_weight: float = 1.0
    margin_weight: float = 0.5
    margin: float = 0.2
    entropy_weight: float = 1.0
    h_tgsd_target_weight: float = 1.0
    h_tgsd_entropy_weight: float = 1.0
    h_tgsd_unrelated_kl_weight: float = 1.0
    h_tgsd_unrelated_feature_weight: float = 1.0
    h_tgsd_sibling_kl_weight: float = 1.0
    h_tgsd_sibling_feature_weight: float = 1.0
    h_tgsd_shared_weight: float = 1.0
    h_tgsd_text_weight: float = 0.5
    h_tgsd_prototype_samples_per_class: int = 200
    h_tgsd_basis_tolerance: float = 1e-5
    train_logit_scale: bool = True


def _sample_counterfactual(labels: torch.Tensor, num_classes: int, rng: random.Random) -> torch.Tensor:
    # Createes new label tensor by replacing each true label with a randomly chosen class.
    y_cf = labels.clone()
    for i in range(labels.size(0)):
        y = int(labels[i].item())
        candidates = [c for c in range(num_classes) if c != y]
        y_cf[i] = int(rng.choice(candidates))
    return y_cf


def _kl_div(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    # Kullback-Leibler divergence -> measures how one probability distribution diverges from another distribution
    # kl_div = 0 : identical distribs
    # kl_div > 0 : different distribs 
    # Temperature Scaling: controls smoothness of distribs
    # temp = {1: Normal Softmax, >1: Softer probabilities, ->infinity: Uniform distribution, <1: Sharper probabilities}
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature**2)


def _eval_snapshot(
    model,
    loaders,
    class_text_inputs,
    device: torch.device,
    non_blocking: bool = False,
) -> Dict[str, float]:
    model.eval()
    class_text_features = build_class_text_features(
        model, class_text_inputs, device
    )
    evaluation_args = {
        "class_text_inputs": class_text_inputs,
        "device": device,
        "class_text_features": class_text_features,
        "non_blocking": non_blocking,
    }
    retain_train = evaluate_classification(
        model, loaders["retain_train"], **evaluation_args
    )
    retain_val = evaluate_classification(
        model, loaders["retain_val"], **evaluation_args
    )
    forget = evaluate_classification(
        model, loaders["forget"], **evaluation_args
    )
    test_retain = evaluate_classification(
        model, loaders["test_retain"], **evaluation_args
    )
    test_all = evaluate_classification(
        model, loaders["test_all"], **evaluation_args
    )
    return {
        "retain_train_acc": retain_train["accuracy"],
        "retain_val_acc": retain_val["accuracy"],
        "forget_acc": forget["accuracy"],
        "test_retain_acc": test_retain["accuracy"],
        "test_all_acc": test_all["accuracy"],
    }


def _mean_component_losses(
    history: Mapping[str, list[float]],
) -> Dict[str, float]:
    return {
        f"loss_{name}": float(sum(values) / len(values))
        for name, values in history.items()
        if values
    }


def _validate_h_tgsd_request(
    split: Mapping[str, object],
) -> tuple[str, list[int], list[int]]:
    request_type = str(split.get("request_type") or "")
    target_classes = [
        int(value)
        for value in split.get(
            "target_classes", split.get("forget_classes", [])
        )
    ]
    sibling_classes = [
        int(value) for value in split.get("sibling_classes", [])
    ]
    if request_type not in {"superclass", "selective_class"}:
        raise ValueError(
            "H-TGSD requires a superclass or selective_class split request"
        )
    if request_type == "selective_class" and not sibling_classes:
        raise ValueError(
            "Selective-class H-TGSD requires sibling classes in split metadata"
        )
    return request_type, target_classes, sibling_classes


def _validate_h_tgsd_config(cfg: UnlearnConfig) -> None:
    if not 0.0 <= cfg.h_tgsd_text_weight <= 1.0:
        raise ValueError("h_tgsd_text_weight must be in [0, 1]")
    if cfg.h_tgsd_prototype_samples_per_class < 1:
        raise ValueError(
            "h_tgsd_prototype_samples_per_class must be at least 1"
        )
    if cfg.h_tgsd_basis_tolerance <= 0:
        raise ValueError("h_tgsd_basis_tolerance must be positive")
    weights = {
        "h_tgsd_target_weight": cfg.h_tgsd_target_weight,
        "h_tgsd_entropy_weight": cfg.h_tgsd_entropy_weight,
        "h_tgsd_unrelated_kl_weight": cfg.h_tgsd_unrelated_kl_weight,
        "h_tgsd_unrelated_feature_weight": (
            cfg.h_tgsd_unrelated_feature_weight
        ),
        "h_tgsd_sibling_kl_weight": cfg.h_tgsd_sibling_kl_weight,
        "h_tgsd_sibling_feature_weight": (
            cfg.h_tgsd_sibling_feature_weight
        ),
        "h_tgsd_shared_weight": cfg.h_tgsd_shared_weight,
    }
    negative = [name for name, value in weights.items() if value < 0]
    if negative:
        raise ValueError(f"H-TGSD weights must be non-negative: {negative}")


def _student_teacher_outputs(
    *,
    model,
    teacher,
    batch: Dict[str, torch.Tensor],
    student_text_features: torch.Tensor,
    teacher_text_features: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    student_features = model.encode_images(batch["pixel_values"])
    student_logits = model.logits_from_embeddings(
        student_features, student_text_features
    )
    with torch.no_grad():
        teacher_features = teacher.encode_images(batch["pixel_values"])
        teacher_logits = teacher.logits_from_embeddings(
            teacher_features, teacher_text_features
        )
    return (
        student_features,
        student_logits,
        teacher_features,
        teacher_logits,
    )


def run_unlearning(cfg: UnlearnConfig) -> Dict[str, str | float]:
    run_started = time.perf_counter()
    if cfg.method not in UNLEARNING_METHODS:
        raise ValueError(f"Unsupported method={cfg.method}. Choose from {sorted(UNLEARNING_METHODS)}")
    split, dataset_spec, class_names = load_split_metadata(
        cfg.split_path, cfg.dataset_name
    )
    h_tgsd_request = None
    if cfg.method in H_TGSD_METHODS:
        _validate_h_tgsd_config(cfg)
        h_tgsd_request = _validate_h_tgsd_request(split)

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    provenance = collect_run_provenance(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    output_dir = Path(cfg.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    image_processor = CLIPImageProcessor.from_pretrained(cfg.model_name)
    tokenizer = CLIPTokenizer.from_pretrained(cfg.model_name)
    class_text_inputs = build_text_inputs(
        tokenizer,
        class_names=class_names,
        template=cfg.prompt_template,
    )
    class_text_inputs = {k: v.to(device) for k, v in class_text_inputs.items()}

    loaders = build_loaders(
        data_dir=cfg.data_dir,
        split_path=cfg.split_path,
        image_processor=image_processor,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        dataset_name=dataset_spec.name,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
    )

    model, finetune_meta = load_checkpoint(cfg.finetuned_checkpoint, map_location=device)
    validate_checkpoint_dataset(
        finetune_meta, dataset_spec.name, cfg.finetuned_checkpoint
    )
    model = model.to(device)
    model.configure_for_unlearning(cfg.train_logit_scale)
    model.clip.eval()

    teacher = copy.deepcopy(model).eval().to(device) # left-to-right chaining, deepcopy doesnt affect original
    for param in teacher.parameters():
        param.requires_grad = False
    teacher_text_features = build_class_text_features(
        teacher, class_text_inputs, device
    )
    student_text_features = None
    if model.can_cache_text_features_during_training():
        student_text_features = build_class_text_features(
            model, class_text_inputs, device
        )

    optimizer = AdamW(model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Infinite iterators to keep yielding batches because retain_train set >> forget 
    forget_iter = cycle_loader(loaders["forget"]) 
    retain_iter = cycle_loader(loaders["retain_train"])
    unrelated_iter = None
    sibling_iter = None
    h_tgsd_bases: SemanticBases | None = None
    h_tgsd_metadata: Dict[str, object] | None = None
    h_tgsd_basis_path: Path | None = None
    if cfg.method in H_TGSD_METHODS:
        if student_text_features is None:
            raise ValueError(
                "H-TGSD requires a frozen text path with cacheable text features"
            )
        if h_tgsd_request is None:
            raise RuntimeError("H-TGSD request was not initialized")
        request_type, target_classes, sibling_classes = h_tgsd_request
        if len(loaders["unrelated_retain"].dataset) == 0:
            raise ValueError("H-TGSD requires unrelated retain samples")
        prototype_loaders = [loaders["forget"]]
        if sibling_classes:
            if len(loaders["sibling_retain"].dataset) == 0:
                raise ValueError(
                    "H-TGSD requires sibling retain samples for this request"
                )
            prototype_loaders.append(loaders["sibling_retain"])
            if cfg.method == "h_tgsd":
                sibling_iter = cycle_loader(loaders["sibling_retain"])
        with precision_context(model.cfg.precision, device):
            prototypes, prototype_counts = collect_class_image_prototypes(
                model=teacher,
                loaders=prototype_loaders,
                class_ids=[*target_classes, *sibling_classes],
                samples_per_class=cfg.h_tgsd_prototype_samples_per_class,
                device=device,
                non_blocking=cfg.non_blocking,
            )
        h_tgsd_bases = build_semantic_bases(
            class_image_prototypes=prototypes,
            class_text_features=teacher_text_features,
            target_classes=target_classes,
            sibling_classes=sibling_classes,
            request_type=request_type,
            text_weight=cfg.h_tgsd_text_weight,
            tolerance=cfg.h_tgsd_basis_tolerance,
        )
        unrelated_iter = cycle_loader(loaders["unrelated_retain"])
        h_tgsd_metadata = {
            **h_tgsd_bases.checkpoint_payload(),
            "target_classes": target_classes,
            "sibling_classes": sibling_classes,
            "prototype_counts": prototype_counts,
            "text_weight": cfg.h_tgsd_text_weight,
            "basis_tolerance": cfg.h_tgsd_basis_tolerance,
        }
        h_tgsd_basis_path = metrics_dir / f"{cfg.method}_basis.pt"
        torch.save(h_tgsd_metadata, h_tgsd_basis_path)
    
    rng = random.Random(cfg.seed)

    losses = []
    component_history: Dict[str, list[float]] = {}
    show_progress = os.environ.get("UNML_TQDM", "0") == "1"

    optimization_started = time.perf_counter()
    for step in trange(cfg.steps, desc=f"unlearn:{cfg.method}", disable=not show_progress):
        model.set_train_mode()

        with precision_context(model.cfg.precision, device):
            if cfg.method in H_TGSD_METHODS:
                if h_tgsd_bases is None or unrelated_iter is None:
                    raise RuntimeError("H-TGSD context was not initialized")
                batch_f = move_to_device(
                    next(forget_iter),
                    device,
                    non_blocking=cfg.non_blocking,
                )
                batch_u = move_to_device(
                    next(unrelated_iter),
                    device,
                    non_blocking=cfg.non_blocking,
                )
                target_outputs = _student_teacher_outputs(
                    model=model,
                    teacher=teacher,
                    batch=batch_f,
                    student_text_features=student_text_features,
                    teacher_text_features=teacher_text_features,
                )
                unrelated_outputs = _student_teacher_outputs(
                    model=model,
                    teacher=teacher,
                    batch=batch_u,
                    student_text_features=student_text_features,
                    teacher_text_features=teacher_text_features,
                )
                sibling_outputs = (None, None, None, None)
                if sibling_iter is not None:
                    batch_s = move_to_device(
                        next(sibling_iter),
                        device,
                        non_blocking=cfg.non_blocking,
                    )
                    sibling_outputs = _student_teacher_outputs(
                        model=model,
                        teacher=teacher,
                        batch=batch_s,
                        student_text_features=student_text_features,
                        teacher_text_features=teacher_text_features,
                    )
                components = h_tgsd_objective(
                    target_features=target_outputs[0],
                    target_teacher_features=target_outputs[2],
                    target_logits=target_outputs[1],
                    unrelated_student_features=unrelated_outputs[0],
                    unrelated_student_logits=unrelated_outputs[1],
                    unrelated_teacher_features=unrelated_outputs[2],
                    unrelated_teacher_logits=unrelated_outputs[3],
                    sibling_student_features=sibling_outputs[0],
                    sibling_student_logits=sibling_outputs[1],
                    sibling_teacher_features=sibling_outputs[2],
                    sibling_teacher_logits=sibling_outputs[3],
                    bases=h_tgsd_bases,
                    temperature=cfg.kl_temperature,
                    target_weight=cfg.h_tgsd_target_weight,
                    entropy_weight=cfg.h_tgsd_entropy_weight,
                    unrelated_kl_weight=cfg.h_tgsd_unrelated_kl_weight,
                    unrelated_feature_weight=(
                        cfg.h_tgsd_unrelated_feature_weight
                    ),
                    sibling_kl_weight=cfg.h_tgsd_sibling_kl_weight,
                    sibling_feature_weight=(
                        cfg.h_tgsd_sibling_feature_weight
                    ),
                    shared_weight=cfg.h_tgsd_shared_weight,
                    preserve_siblings=(
                        cfg.method != "h_tgsd_no_sibling_preservation"
                    ),
                )
                loss = components["total"]
            else:
                batch_r = move_to_device(
                    next(retain_iter),
                    device,
                    non_blocking=cfg.non_blocking,
                )
                if student_text_features is None:
                    logits_r = model.class_logits(
                        pixel_values=batch_r["pixel_values"],
                        class_input_ids=class_text_inputs["input_ids"],
                        class_attention_mask=class_text_inputs[
                            "attention_mask"
                        ],
                    )
                else:
                    logits_r = model.class_logits_from_text_features(
                        pixel_values=batch_r["pixel_values"],
                        class_text_features=student_text_features,
                    )
                if cfg.method == "retain_only":
                    loss = F.cross_entropy(logits_r, batch_r["labels"])
                else:
                    with torch.no_grad():
                        t_logits_r = teacher.class_logits_from_text_features(
                            pixel_values=batch_r["pixel_values"],
                            class_text_features=teacher_text_features,
                        )
                    retain_kl = _kl_div(
                        logits_r, t_logits_r, cfg.kl_temperature
                    )
                    batch_f = move_to_device(
                        next(forget_iter),
                        device,
                        non_blocking=cfg.non_blocking,
                    )
                    if student_text_features is None:
                        logits_f = model.class_logits(
                            pixel_values=batch_f["pixel_values"],
                            class_input_ids=class_text_inputs["input_ids"],
                            class_attention_mask=class_text_inputs[
                                "attention_mask"
                            ],
                        )
                    else:
                        logits_f = model.class_logits_from_text_features(
                            pixel_values=batch_f["pixel_values"],
                            class_text_features=student_text_features,
                        )

            if cfg.method in H_TGSD_METHODS or cfg.method == "retain_only":
                pass
            elif cfg.method == "ga_kl":
                # Gradient Ascent (GA) with KL divergence:
                # GA to maximize loss on the forget data while using KL div to make sure the model doesnt stray from the teacher
                forget_term = -F.cross_entropy(logits_f, batch_f["labels"]) # negation reverses the minimization and model is encouraged to become worse at predicting 
                loss = cfg.ga_weight * forget_term + cfg.kl_weight * retain_kl # add retain tensor to ensure retained behaviour close to teacher

            elif cfg.method == "counterfactual_rebind":
                # Redirects forget predictions to a randomly chosen wrong class (y_cf),
                # then enforces a margin so cf score > true score.
                # WEAKNESS: both rebind_loss and margin_loss saturate at high step counts —
                # once the model confidently predicts y_cf, gradients from these terms
                # shrink to ~0 and only retain_kl remains, making it behave like ga_kl.
                y_true = batch_f["labels"]
                y_cf = _sample_counterfactual(y_true, len(class_names), rng).to(device)
                rebind_loss = F.cross_entropy(logits_f, y_cf) # Redirect to counterfactual
                true_logits = logits_f.gather(1, y_true.unsqueeze(1)).squeeze(1) # score of the true class
                cf_logits = logits_f.gather(1, y_cf.unsqueeze(1)).squeeze(1) # score of the counterfactual class
                margin_loss = F.relu(true_logits - cf_logits + cfg.margin).mean() # penalizes the model when true score > counterfactual score

                loss = (
                    cfg.cf_weight * rebind_loss
                    + cfg.margin_weight * margin_loss
                    + cfg.kl_weight * retain_kl
                )

            elif cfg.method == "entropy_rebind":
                # Instead of pushing toward a single counterfactual class, maximizes entropy
                # over ALL classes on forget data — the model becomes maximally uncertain.
                # KEY DIFFERENCE from counterfactual_rebind: entropy never saturates.
                # No matter how many steps, there is always gradient pulling each class logit
                # toward equal probability (1/C). The model can never be "done" forgetting.
                # No margin loss needed: entropy maximization inherently suppresses the true
                # class logit as part of flattening the whole distribution.
                log_probs_f = F.log_softmax(logits_f, dim=-1)
                probs_f = log_probs_f.exp()
                entropy = -(probs_f * log_probs_f).sum(dim=-1).mean()  # H(p), higher = more uncertain
                loss = -cfg.entropy_weight * entropy + cfg.kl_weight * retain_kl  # negate to maximise

            else:
                raise RuntimeError(f"Unreachable method: {cfg.method}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), max_norm=1.0)
        optimizer.step()

        losses.append(tensor_to_float(loss))
        if cfg.method in H_TGSD_METHODS:
            for name, value in components.items():
                component_history.setdefault(name, []).append(
                    tensor_to_float(value)
                )

        if (step + 1) % max(10, cfg.steps // 5) == 0:
            recent_loss = sum(losses[-10:]) / min(10, len(losses))
            component_text = ""
            if cfg.method in H_TGSD_METHODS:
                component_text = " " + format_metrics(
                    {
                        name: values[-1]
                        for name, values in component_history.items()
                        if name != "total"
                    }
                )
            print(
                f"[unlearn:{cfg.method}] step={step+1} "
                f"loss={recent_loss:.4f}{component_text}",
                flush=True,
            )
    optimization_seconds = time.perf_counter() - optimization_started

    model.eval()
    evaluation_started = time.perf_counter()
    summary_metrics = _eval_snapshot(
        model,
        loaders,
        class_text_inputs,
        device,
        non_blocking=cfg.non_blocking,
    )
    evaluation_seconds = time.perf_counter() - evaluation_started
    summary_metrics["avg_train_loss"] = float(sum(losses) / max(1, len(losses)))
    summary_metrics["steps"] = float(cfg.steps)
    summary_metrics.update(_mean_component_losses(component_history))
    if h_tgsd_bases is not None:
        summary_metrics["target_basis_rank"] = float(
            h_tgsd_bases.target.shape[1]
        )
        summary_metrics["shared_basis_rank"] = float(
            h_tgsd_bases.shared.shape[1]
        )
    print(f"[unlearn:{cfg.method}] {format_metrics(summary_metrics)}", flush=True)

    ckpt_path = ckpt_dir / f"unlearn_{cfg.method}.pt"
    runtime = {
        "optimization_seconds": optimization_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": time.perf_counter() - run_started,
        "steps": cfg.steps,
        "steps_per_second": (
            cfg.steps / optimization_seconds
            if optimization_seconds > 0
            else 0.0
        ),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        ),
    }
    architecture = model.architecture_summary()
    save_checkpoint(
        str(ckpt_path),
        model,
        extra={
            "stage": "unlearned",
            "method": cfg.method,
            "dataset": dataset_spec.name,
            "class_names": class_names,
            "finetuned_checkpoint": cfg.finetuned_checkpoint,
            "finetune_meta": finetune_meta,
            "architecture": architecture,
            "provenance": provenance,
            "runtime": runtime,
            "unlearning_config": cfg.__dict__,
            "cached_student_text_features": (
                student_text_features is not None
            ),
            "h_tgsd_basis": h_tgsd_metadata,
            "metrics": summary_metrics,
        },
    )
    runtime["checkpoint_size_bytes"] = ckpt_path.stat().st_size

    metrics_path = metrics_dir / f"unlearn_{cfg.method}_metrics.json"
    save_json(
        {
            "method": cfg.method,
            "summary_metrics": summary_metrics,
            "config": cfg.__dict__,
            "architecture": architecture,
            "provenance": provenance,
            "runtime": runtime,
            "basis_path": (
                str(h_tgsd_basis_path)
                if h_tgsd_basis_path is not None
                else None
            ),
        },
        metrics_path,
    )

    log_unlearn_run(cfg.__dict__, summary_metrics)

    result: Dict[str, str | float] = {
        "checkpoint": str(ckpt_path),
        "metrics_path": str(metrics_path),
        **summary_metrics,
    }
    if h_tgsd_basis_path is not None:
        result["basis_path"] = str(h_tgsd_basis_path)
    return result
