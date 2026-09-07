from __future__ import annotations

import pytest
import torch

from unml.disentangle import (
    SemanticBases,
    build_semantic_bases,
    collect_class_image_prototypes,
    h_tgsd_objective,
    orthonormal_basis,
)


def test_orthonormal_basis_spans_input_vectors() -> None:
    vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )

    basis = orthonormal_basis(vectors)

    assert basis.shape == (3, 2)
    assert torch.allclose(
        basis.t() @ basis,
        torch.eye(2),
        atol=1e-6,
    )
    reconstructed = (vectors @ basis) @ basis.t()
    assert torch.allclose(reconstructed, vectors, atol=1e-6)


def test_selective_basis_separates_target_residual_from_shared_space() -> None:
    image_prototypes = {
        0: torch.tensor([1.0, 0.0, 1.0]),
        1: torch.tensor([1.0, 0.0, 0.0]),
        2: torch.tensor([0.0, 1.0, 0.0]),
    }
    text_features = torch.stack(
        [
            torch.tensor([1.0, 0.0, 1.0]),
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
        ]
    )

    bases = build_semantic_bases(
        class_image_prototypes=image_prototypes,
        class_text_features=text_features,
        target_classes=[0],
        sibling_classes=[1, 2],
        request_type="selective_class",
        text_weight=0.5,
    )

    assert bases.target.shape == (3, 1)
    assert bases.shared.shape == (3, 2)
    assert torch.allclose(
        bases.target.t() @ bases.shared,
        torch.zeros((1, 2)),
        atol=1e-6,
    )


class IdentityImageModel:
    def eval(self):
        return self

    def encode_images(self, pixel_values):
        return pixel_values


def test_collect_class_prototypes_honors_per_class_limit() -> None:
    batches = [
        {
            "pixel_values": torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ]
            ),
            "labels": torch.tensor([0, 1, 0]),
            "indices": torch.tensor([0, 1, 2]),
        }
    ]

    prototypes, counts = collect_class_image_prototypes(
        model=IdentityImageModel(),
        loaders=[batches],
        class_ids=[0, 1],
        samples_per_class=1,
        device=torch.device("cpu"),
    )

    assert counts == {0: 1, 1: 1}
    assert torch.allclose(prototypes[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(prototypes[1], torch.tensor([0.0, 1.0]))


def test_h_tgsd_objective_has_component_gradients_and_ablation() -> None:
    target_features = torch.tensor(
        [[0.0, 0.0, 1.0]], requires_grad=True
    )
    target_logits = torch.tensor(
        [[3.0, 0.0, -1.0]], requires_grad=True
    )
    unrelated_features = torch.tensor(
        [[0.8, 0.2, 0.0]], requires_grad=True
    )
    unrelated_teacher_features = torch.tensor([[1.0, 0.0, 0.0]])
    unrelated_logits = torch.tensor(
        [[2.0, 0.0, -1.0]], requires_grad=True
    )
    unrelated_teacher_logits = torch.tensor([[2.5, -0.5, -1.0]])
    sibling_features = torch.tensor(
        [[0.7, 0.3, 0.0]], requires_grad=True
    )
    sibling_teacher_features = torch.tensor([[1.0, 0.0, 0.0]])
    sibling_logits = torch.tensor(
        [[1.5, 0.5, -1.0]], requires_grad=True
    )
    sibling_teacher_logits = torch.tensor([[2.0, 0.0, -1.0]])
    bases = SemanticBases(
        target=torch.tensor([[0.0], [0.0], [1.0]]),
        shared=torch.tensor([[1.0], [0.0], [0.0]]),
        semantic_directions={},
        request_type="selective_class",
    )

    components = h_tgsd_objective(
        target_features=target_features,
        target_teacher_features=torch.tensor([[1.0, 0.0, 1.0]]),
        target_logits=target_logits,
        unrelated_student_features=unrelated_features,
        unrelated_teacher_features=unrelated_teacher_features,
        unrelated_student_logits=unrelated_logits,
        unrelated_teacher_logits=unrelated_teacher_logits,
        sibling_student_features=sibling_features,
        sibling_teacher_features=sibling_teacher_features,
        sibling_student_logits=sibling_logits,
        sibling_teacher_logits=sibling_teacher_logits,
        bases=bases,
        temperature=1.5,
        target_weight=1.0,
        entropy_weight=1.0,
        unrelated_kl_weight=1.0,
        unrelated_feature_weight=1.0,
        sibling_kl_weight=1.0,
        sibling_feature_weight=1.0,
        shared_weight=1.0,
    )
    components["total"].backward()

    assert components["target_suppression"].item() == pytest.approx(1.0)
    assert target_features.grad is not None
    assert target_logits.grad is not None
    assert unrelated_features.grad is not None
    assert sibling_features.grad is not None

    ablated = h_tgsd_objective(
        target_features=target_features.detach(),
        target_teacher_features=torch.tensor([[1.0, 0.0, 1.0]]),
        target_logits=target_logits.detach(),
        unrelated_student_features=unrelated_features.detach(),
        unrelated_teacher_features=unrelated_teacher_features,
        unrelated_student_logits=unrelated_logits.detach(),
        unrelated_teacher_logits=unrelated_teacher_logits,
        sibling_student_features=sibling_features.detach(),
        sibling_teacher_features=sibling_teacher_features,
        sibling_student_logits=sibling_logits.detach(),
        sibling_teacher_logits=sibling_teacher_logits,
        bases=bases,
        temperature=1.5,
        target_weight=1.0,
        entropy_weight=1.0,
        unrelated_kl_weight=1.0,
        unrelated_feature_weight=1.0,
        sibling_kl_weight=1.0,
        sibling_feature_weight=1.0,
        shared_weight=1.0,
        preserve_siblings=False,
    )
    assert ablated["sibling_prediction_kl"].item() == 0.0
    assert ablated["sibling_feature_preservation"].item() == 0.0
    assert ablated["shared_subspace_preservation"].item() == 0.0
