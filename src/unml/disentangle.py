from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import move_to_device


@dataclass(frozen=True)
class SemanticBases:
    target: torch.Tensor
    shared: torch.Tensor
    semantic_directions: Mapping[int, torch.Tensor]
    request_type: str

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "request_type": self.request_type,
            "target": self.target.detach().cpu(),
            "shared": self.shared.detach().cpu(),
            "semantic_directions": {
                str(class_id): direction.detach().cpu()
                for class_id, direction in self.semantic_directions.items()
            },
        }


def orthonormal_basis(
    vectors: torch.Tensor,
    tolerance: float = 1e-5,
) -> torch.Tensor:
    """Return a column-oriented orthonormal basis for row vectors."""
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [count, dimension]")
    if vectors.shape[0] == 0:
        return vectors.new_zeros((vectors.shape[1], 0))
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    _, singular_values, vh = torch.linalg.svd(
        vectors.float(), full_matrices=False
    )
    if singular_values.numel() == 0:
        return vectors.new_zeros((vectors.shape[1], 0))
    threshold = tolerance * singular_values.max()
    rank = int((singular_values > threshold).sum().item())
    if rank == 0:
        raise ValueError("Cannot construct a basis from zero-valued vectors")
    return vh[:rank].t().to(device=vectors.device, dtype=vectors.dtype)


def project_onto_basis(
    features: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    if features.ndim != 2 or basis.ndim != 2:
        raise ValueError("features and basis must both be matrices")
    if features.shape[1] != basis.shape[0]:
        raise ValueError("Feature and basis dimensions do not match")
    return (features @ basis) @ basis.t()


def subspace_energy(
    features: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    if basis.shape[1] == 0:
        return features.new_zeros(())
    coordinates = features.float() @ basis.float()
    return coordinates.square().sum(dim=-1).mean()


def _normalize_direction(value: torch.Tensor, name: str) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value.float())
    if not torch.isfinite(norm) or float(norm.item()) <= 1e-12:
        raise ValueError(f"{name} has zero or non-finite norm")
    return F.normalize(value.float(), dim=0).to(dtype=value.dtype)


def build_semantic_bases(
    *,
    class_image_prototypes: Mapping[int, torch.Tensor],
    class_text_features: torch.Tensor,
    target_classes: Sequence[int],
    sibling_classes: Sequence[int],
    request_type: str,
    text_weight: float = 0.5,
    tolerance: float = 1e-5,
) -> SemanticBases:
    if not 0.0 <= text_weight <= 1.0:
        raise ValueError("text_weight must be in [0, 1]")
    targets = sorted(set(int(value) for value in target_classes))
    siblings = sorted(set(int(value) for value in sibling_classes))
    if not targets:
        raise ValueError("At least one target class is required")
    if set(targets) & set(siblings):
        raise ValueError("Target and sibling classes must be disjoint")
    required = sorted(set(targets + siblings))
    missing = [
        class_id
        for class_id in required
        if class_id not in class_image_prototypes
    ]
    if missing:
        raise ValueError(f"Missing image prototypes for classes: {missing}")

    semantic_directions = {}
    for class_id in required:
        image_direction = _normalize_direction(
            class_image_prototypes[class_id],
            f"image prototype {class_id}",
        )
        text_direction = _normalize_direction(
            class_text_features[class_id],
            f"text feature {class_id}",
        )
        combined = (
            (1.0 - text_weight) * image_direction
            + text_weight * text_direction
        )
        semantic_directions[class_id] = _normalize_direction(
            combined, f"semantic direction {class_id}"
        )

    reference = next(iter(semantic_directions.values()))
    if request_type == "superclass":
        target_basis = orthonormal_basis(
            torch.stack([semantic_directions[value] for value in targets]),
            tolerance=tolerance,
        )
        shared_basis = reference.new_zeros((reference.numel(), 0))
    elif request_type == "selective_class":
        if len(targets) != 1:
            raise ValueError(
                "Selective-class H-TGSD requires exactly one target class"
            )
        if not siblings:
            raise ValueError(
                "Selective-class H-TGSD requires sibling classes"
            )
        shared_basis = orthonormal_basis(
            torch.stack([semantic_directions[value] for value in siblings]),
            tolerance=tolerance,
        )
        target_direction = semantic_directions[targets[0]].unsqueeze(0)
        residual = target_direction - project_onto_basis(
            target_direction, shared_basis
        )
        target_basis = orthonormal_basis(residual, tolerance=tolerance)
    else:
        raise ValueError(
            "H-TGSD supports request_type superclass or selective_class"
        )

    return SemanticBases(
        target=target_basis,
        shared=shared_basis,
        semantic_directions=semantic_directions,
        request_type=request_type,
    )


@torch.no_grad()
def collect_class_image_prototypes(
    *,
    model,
    loaders: Iterable[DataLoader],
    class_ids: Sequence[int],
    samples_per_class: int,
    device: torch.device,
    non_blocking: bool = False,
) -> tuple[dict[int, torch.Tensor], dict[int, int]]:
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be at least 1")
    requested = sorted(set(int(value) for value in class_ids))
    if not requested:
        raise ValueError("At least one prototype class is required")

    sums: dict[int, torch.Tensor] = {}
    counts = {class_id: 0 for class_id in requested}
    model.eval()
    for loader in loaders:
        for batch in loader:
            batch = move_to_device(
                batch, device, non_blocking=non_blocking
            )
            features = model.encode_images(batch["pixel_values"]).float()
            for class_id in requested:
                remaining = samples_per_class - counts[class_id]
                if remaining <= 0:
                    continue
                selected = features[batch["labels"] == class_id][:remaining]
                if selected.numel() == 0:
                    continue
                class_sum = selected.sum(dim=0)
                sums[class_id] = sums.get(class_id, torch.zeros_like(class_sum))
                sums[class_id] += class_sum
                counts[class_id] += int(selected.shape[0])
            if all(count >= samples_per_class for count in counts.values()):
                break
        if all(count >= samples_per_class for count in counts.values()):
            break

    missing = [class_id for class_id, count in counts.items() if count == 0]
    if missing:
        raise ValueError(
            f"No prototype samples were found for classes: {missing}"
        )
    prototypes = {
        class_id: F.normalize(
            sums[class_id] / counts[class_id], dim=0
        )
        for class_id in requested
    }
    return prototypes, counts


def prediction_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher = F.softmax(teacher_logits.float() / temperature, dim=-1)
    return (
        F.kl_div(student, teacher, reduction="batchmean")
        * temperature**2
    )


def feature_preservation(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
) -> torch.Tensor:
    return (
        1.0
        - F.cosine_similarity(
            student_features.float(),
            teacher_features.float(),
            dim=-1,
        )
    ).mean()


def shared_coordinate_preservation(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    shared_basis: torch.Tensor,
) -> torch.Tensor:
    if shared_basis.shape[1] == 0:
        return student_features.new_zeros(())
    return F.mse_loss(
        student_features.float() @ shared_basis.float(),
        teacher_features.float() @ shared_basis.float(),
    )


def h_tgsd_objective(
    *,
    target_features: torch.Tensor,
    target_teacher_features: torch.Tensor,
    target_logits: torch.Tensor,
    unrelated_student_features: torch.Tensor,
    unrelated_teacher_features: torch.Tensor,
    unrelated_student_logits: torch.Tensor,
    unrelated_teacher_logits: torch.Tensor,
    bases: SemanticBases,
    temperature: float,
    target_weight: float,
    entropy_weight: float,
    unrelated_kl_weight: float,
    unrelated_feature_weight: float,
    sibling_student_features: torch.Tensor | None = None,
    sibling_teacher_features: torch.Tensor | None = None,
    sibling_student_logits: torch.Tensor | None = None,
    sibling_teacher_logits: torch.Tensor | None = None,
    sibling_kl_weight: float = 0.0,
    sibling_feature_weight: float = 0.0,
    shared_weight: float = 0.0,
    preserve_siblings: bool = True,
) -> dict[str, torch.Tensor]:
    log_probabilities = F.log_softmax(target_logits.float(), dim=-1)
    probabilities = log_probabilities.exp()
    entropy = -(probabilities * log_probabilities).sum(dim=-1).mean()
    components = {
        "target_suppression": subspace_energy(
            target_features, bases.target
        ),
        "forget_entropy": -entropy,
        "unrelated_prediction_kl": prediction_kl(
            unrelated_student_logits,
            unrelated_teacher_logits,
            temperature,
        ),
        "unrelated_feature_preservation": feature_preservation(
            unrelated_student_features,
            unrelated_teacher_features,
        ),
        "sibling_prediction_kl": target_features.new_zeros(()),
        "sibling_feature_preservation": target_features.new_zeros(()),
        "shared_subspace_preservation": target_features.new_zeros(()),
    }
    has_sibling_batch = all(
        value is not None
        for value in (
            sibling_student_features,
            sibling_teacher_features,
            sibling_student_logits,
            sibling_teacher_logits,
        )
    )
    shared_losses = []
    if preserve_siblings and bases.shared.shape[1] > 0:
        shared_losses.append(
            shared_coordinate_preservation(
                target_features,
                target_teacher_features,
                bases.shared,
            )
        )
    if preserve_siblings and has_sibling_batch:
        components["sibling_prediction_kl"] = prediction_kl(
            sibling_student_logits,
            sibling_teacher_logits,
            temperature,
        )
        components["sibling_feature_preservation"] = feature_preservation(
            sibling_student_features,
            sibling_teacher_features,
        )
        shared_losses.append(
            shared_coordinate_preservation(
                sibling_student_features,
                sibling_teacher_features,
                bases.shared,
            )
        )
    if shared_losses:
        components["shared_subspace_preservation"] = torch.stack(
            shared_losses
        ).mean()

    components["total"] = (
        target_weight * components["target_suppression"]
        + entropy_weight * components["forget_entropy"]
        + unrelated_kl_weight * components["unrelated_prediction_kl"]
        + unrelated_feature_weight
        * components["unrelated_feature_preservation"]
        + sibling_kl_weight * components["sibling_prediction_kl"]
        + sibling_feature_weight
        * components["sibling_feature_preservation"]
        + shared_weight * components["shared_subspace_preservation"]
    )
    return components
