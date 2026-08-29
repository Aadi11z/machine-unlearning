from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unml import data
from unml.utils import move_to_device


class TinyDataset(torch.utils.data.Dataset):
    targets = [0, 1]

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return torch.zeros(1), self.targets[index], index


class IdentityProcessor:
    size = {"shortest_edge": 1}
    crop_size = {"height": 1}
    image_mean = [0.0]
    image_std = [1.0]


def _patch_loader_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    split = {
        "forget_indices": [0],
        "retain_train_indices": [1],
        "retain_val_indices": [1],
        "finetune_train_indices": [0, 1],
        "test_forget_indices": [0],
        "test_retain_indices": [1],
        "test_all_indices": [0, 1],
    }
    spec = SimpleNamespace(name="cifar10")
    monkeypatch.setattr(
        data,
        "load_split_metadata",
        lambda *_args, **_kwargs: (split, spec, []),
    )
    monkeypatch.setattr(
        data,
        "_load_dataset_pair",
        lambda *_args, **_kwargs: (TinyDataset(), TinyDataset()),
    )
    monkeypatch.setattr(data, "make_collate_fn", lambda _processor: lambda batch: batch)


def test_build_loaders_disables_worker_only_options_at_zero_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_loader_inputs(monkeypatch)

    loaders = data.build_loaders(
        data_dir="unused",
        split_path="unused",
        image_processor=IdentityProcessor(),
        batch_size=2,
        num_workers=0,
        persistent_workers=True,
        prefetch_factor=4,
    )

    loader = loaders["finetune_train"]
    assert loader.num_workers == 0
    assert not loader.persistent_workers
    assert loader.prefetch_factor is None


def test_build_loaders_applies_worker_runtime_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_loader_inputs(monkeypatch)

    loaders = data.build_loaders(
        data_dir="unused",
        split_path="unused",
        image_processor=IdentityProcessor(),
        batch_size=2,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3,
    )

    loader = loaders["finetune_train"]
    assert loader.pin_memory
    assert loader.persistent_workers
    assert loader.prefetch_factor == 3
    assert not loaders["test_all"].persistent_workers


def test_build_loaders_forwards_canonical_final_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_loader_inputs(monkeypatch)
    calls = []
    monkeypatch.setattr(
        data,
        "load_split_metadata",
        lambda *_args, **kwargs: (calls.append(kwargs) or (
            {
                "forget_indices": [0],
                "retain_train_indices": [1],
                "retain_val_indices": [1],
                "finetune_train_indices": [0, 1],
                "test_forget_indices": [0],
                "test_retain_indices": [1],
                "test_all_indices": [0, 1],
            },
            SimpleNamespace(name="cifar10"),
            [],
        )),
    )

    data.build_loaders(
        data_dir="unused",
        split_path="unused",
        image_processor=IdentityProcessor(),
        batch_size=2,
        num_workers=0,
        canonical_final_fit=True,
    )

    assert calls == [{"canonical_final_fit": True}]


@pytest.mark.parametrize(
    ("num_workers", "prefetch_factor", "message"),
    [(-1, 2, "num_workers"), (1, 0, "prefetch_factor")],
)
def test_build_loaders_rejects_invalid_worker_settings(
    num_workers: int,
    prefetch_factor: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        data.build_loaders(
            data_dir="unused",
            split_path="unused",
            image_processor=IdentityProcessor(),
            batch_size=2,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )


def test_move_to_device_forwards_non_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.tensor([1])
    calls = []
    original_to = torch.Tensor.to

    def recording_to(self, *args, **kwargs):
        calls.append(kwargs.get("non_blocking"))
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", recording_to)

    moved = move_to_device(
        {"value": tensor},
        torch.device("cpu"),
        non_blocking=True,
    )

    assert calls == [True]
    assert moved["value"].device.type == "cpu"
