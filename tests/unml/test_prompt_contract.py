from __future__ import annotations

import pytest
import torch

from unml.prompts import (
    CIFAR100_OPENAI_TEMPLATES,
    CIFAR100_PROMPT_CONTRACT,
    PromptEnsembleContract,
    aggregate_prompt_features,
    encode_prompt_ensemble,
    render_class_prompts,
    resolve_prompt_contract,
    tokenize_prompt_ensemble,
)


class RecordingTokenizer:
    def __init__(self) -> None:
        self.prompts: tuple[str, ...] | None = None

    def __call__(self, texts, **_kwargs):
        self.prompts = tuple(texts)
        rows = len(texts)
        return {
            "input_ids": torch.arange(rows * 3, dtype=torch.long).reshape(rows, 3),
            "attention_mask": torch.ones((rows, 3), dtype=torch.long),
        }


class MatrixTextEncoder:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values

    def encode_text(self, input_ids, attention_mask):
        assert input_ids.shape == attention_mask.shape
        return self.values.to(input_ids.device)


def test_cifar100_contract_matches_openai_18_template_source() -> None:
    contract = resolve_prompt_contract("CIFAR-100")

    assert contract is CIFAR100_PROMPT_CONTRACT
    assert len(contract.templates) == 18
    assert contract.templates == CIFAR100_OPENAI_TEMPLATES
    assert contract.version == "openai_cifar100_v1"
    assert contract.digest == "67941c4387e302ce64242ab21578bc7f83b266acb02e4d0479c102a874aa6cdb"


def test_contract_digest_is_stable_and_order_sensitive() -> None:
    recreated = PromptEnsembleContract(
        dataset_name="CIFAR100",
        version="openai_cifar100_v1",
        templates=CIFAR100_OPENAI_TEMPLATES,
    )
    reordered = PromptEnsembleContract(
        dataset_name="cifar100",
        version="openai_cifar100_v1",
        templates=tuple(reversed(CIFAR100_OPENAI_TEMPLATES)),
    )

    assert recreated.digest == CIFAR100_PROMPT_CONTRACT.digest
    assert reordered.digest != CIFAR100_PROMPT_CONTRACT.digest


def test_render_and_tokenize_are_class_major_template_minor() -> None:
    tokenizer = RecordingTokenizer()
    contract = PromptEnsembleContract(
        dataset_name="cifar100", version="test_v1", templates=("x {}", "y {}")
    )

    prompts = render_class_prompts(("rose", "tulip"), contract)
    inputs = tokenize_prompt_ensemble(tokenizer, ("rose", "tulip"), contract)

    assert prompts == ("x rose", "y rose", "x tulip", "y tulip")
    assert tokenizer.prompts == prompts
    assert inputs.input_ids.shape == (4, 3)
    assert inputs.class_count == 2
    assert inputs.template_count == 2
    assert inputs.contract.digest == contract.digest


def test_feature_aggregation_normalizes_before_and_after_mean() -> None:
    features = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 4.0]],
            [[-5.0, 0.0], [-10.0, 0.0]],
        ]
    )

    aggregated = aggregate_prompt_features(features)

    expected = torch.tensor(
        [[2**-0.5, 2**-0.5], [-1.0, 0.0]], dtype=torch.float32
    )
    assert torch.allclose(aggregated, expected, atol=1e-6)
    assert torch.allclose(
        torch.linalg.vector_norm(aggregated, dim=-1), torch.ones(2)
    )


def test_encoder_uses_contract_dimensions_for_aggregation() -> None:
    contract = PromptEnsembleContract(
        dataset_name="cifar100", version="test_v1", templates=("{} one", "{} two")
    )
    inputs = tokenize_prompt_ensemble(RecordingTokenizer(), ("a", "b"), contract)
    model = MatrixTextEncoder(
        torch.tensor([[2.0, 0.0], [0.0, 2.0], [-3.0, 0.0], [-3.0, 0.0]])
    )

    features = encode_prompt_ensemble(model, inputs, torch.device("cpu"))

    assert features.shape == (2, 2)
    assert torch.allclose(
        features,
        torch.tensor([[2**-0.5, 2**-0.5], [-1.0, 0.0]], dtype=torch.float32),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "dataset_name", ["cifar10", "imagenet", ""]
)
def test_only_cifar100_has_a_canonical_prompt_contract(dataset_name: str) -> None:
    with pytest.raises(ValueError, match="No canonical prompt ensemble"):
        resolve_prompt_contract(dataset_name)


def test_contract_and_feature_validation_rejects_ambiguous_inputs() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PromptEnsembleContract("cifar100", "bad", ("no slot",))
    with pytest.raises(ValueError, match="zero-norm"):
        aggregate_prompt_features(torch.zeros((1, 1, 2)))
    with pytest.raises(ValueError, match="shape"):
        aggregate_prompt_features(torch.ones((1, 2)))
