from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import InterpolationMode

from .config import normalize_dataset_name
from .utils import DEFAULT_PROMPT_TEMPLATE, ensure_dir, load_json, save_json

if TYPE_CHECKING:
    from transformers import CLIPImageProcessor, CLIPTokenizer


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CIFAR100_CLASSES = (
    "apple",
    "aquarium fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak tree",
    "orange",
    "orchid",
    "otter",
    "palm tree",
    "pear",
    "pickup truck",
    "pine tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow tree",
    "wolf",
    "woman",
    "worm",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_class: type[Dataset]
    class_names: tuple[str, ...]


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec("cifar10", CIFAR10, CIFAR10_CLASSES),
    "cifar100": DatasetSpec("cifar100", CIFAR100, CIFAR100_CLASSES),
}


@dataclass
class SplitConfig:
    forget_classes: Sequence[int]
    forget_fraction: float
    retain_val_fraction: float
    seed: int


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    return DATASET_REGISTRY[normalize_dataset_name(dataset_name)]


def load_split_metadata(
    split_path: str, requested_dataset: str | None = None
) -> tuple[Dict[str, object], DatasetSpec, list[str]]:
    split = load_json(split_path)
    stored_dataset = normalize_dataset_name(str(split.get("dataset", "cifar10")))
    if (
        requested_dataset is not None
        and normalize_dataset_name(requested_dataset) != stored_dataset
    ):
        raise ValueError(
            f"Split dataset={stored_dataset!r} does not match "
            f"requested dataset={normalize_dataset_name(requested_dataset)!r}"
        )

    spec = get_dataset_spec(stored_dataset)
    class_names = [str(name) for name in split.get("class_names", spec.class_names)]
    if len(class_names) != len(spec.class_names):
        raise ValueError(
            f"Split class_names has {len(class_names)} entries; "
            f"{stored_dataset} requires {len(spec.class_names)}"
        )
    return split, spec, class_names


def validate_checkpoint_dataset(
    metadata: Mapping[str, object],
    dataset_name: str,
    checkpoint_path: str,
) -> None:
    stored_dataset = metadata.get("dataset")
    if stored_dataset is None:
        return

    expected = normalize_dataset_name(dataset_name)
    actual = normalize_dataset_name(str(stored_dataset))
    if actual != expected:
        raise ValueError(
            f"Checkpoint {checkpoint_path} was created for {actual}, "
            f"but this run uses {expected}"
        )


class CIFARSubset(Dataset):
    def __init__(self, base: Dataset, indices: Sequence[int]):
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = self.indices[item]
        image, label = self.base[idx]
        return image, int(label), int(idx)


class CLIPCollator:
    def __init__(self, image_processor):
        self.image_processor = image_processor
        size_cfg = image_processor.size
        crop_cfg = image_processor.crop_size

        if isinstance(size_cfg, dict):
            resize_size = int(size_cfg.get("shortest_edge") or 224)
        else:
            resize_size = int(size_cfg)

        if isinstance(crop_cfg, dict):
            crop_size = int(crop_cfg.get("height") or resize_size)
        else:
            crop_size = resize_size

        self.transform = T.Compose(
            [
                T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
                T.CenterCrop(crop_size),
                T.ToTensor(),
                T.Normalize(
                    mean=list(image_processor.image_mean),
                    std=list(image_processor.image_std),
                ),
            ]
        )

    def __call__(self, batch):
        images = [sample[0] for sample in batch]
        labels = torch.tensor([sample[1] for sample in batch], dtype=torch.long)
        indices = torch.tensor([sample[2] for sample in batch], dtype=torch.long)
        pixel_values = torch.stack(
            [self.transform(image.convert("RGB")) for image in images], dim=0
        )
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "indices": indices,
        }


def _sample_indices(
    indices: List[int], fraction: float, rng: random.Random
) -> List[int]:
    if fraction >= 0.999:
        return list(indices)
    count = max(1, int(len(indices) * fraction))
    return rng.sample(indices, count)


def make_splits(
    train_labels: Sequence[int],
    test_labels: Sequence[int],
    cfg: SplitConfig,
) -> Dict[str, List[int]]:
    rng = random.Random(cfg.seed)
    forget_pool = [
        i for i, label in enumerate(train_labels) if label in cfg.forget_classes
    ]
    forget_indices = sorted(_sample_indices(forget_pool, cfg.forget_fraction, rng))
    forget_set = set(forget_indices)
    retain_all = [i for i in range(len(train_labels)) if i not in forget_set]
    retain_val_size = max(1, int(len(retain_all) * cfg.retain_val_fraction))
    retain_val_indices = sorted(rng.sample(retain_all, retain_val_size))
    retain_val_set = set(retain_val_indices)
    retain_train_indices = sorted(
        i for i in retain_all if i not in retain_val_set
    )

    return {
        "forget_indices": forget_indices,
        "retain_train_indices": retain_train_indices,
        "retain_val_indices": retain_val_indices,
        "finetune_train_indices": sorted(retain_train_indices + forget_indices),
        "test_forget_indices": [
            i for i, label in enumerate(test_labels) if label in cfg.forget_classes
        ],
        "test_retain_indices": [
            i for i, label in enumerate(test_labels) if label not in cfg.forget_classes
        ],
        "test_all_indices": list(range(len(test_labels))),
        "forget_classes": list(cfg.forget_classes),
    }


def _load_dataset_pair(
    data_dir: str, dataset_name: str, download: bool
) -> tuple[Dataset, Dataset]:
    spec = get_dataset_spec(dataset_name)
    train_ds = spec.dataset_class(root=data_dir, train=True, download=download)
    test_ds = spec.dataset_class(root=data_dir, train=False, download=download)
    return train_ds, test_ds


def download_and_prepare_splits(
    data_dir: str,
    split_path: str,
    forget_classes: Sequence[int],
    forget_fraction: float,
    retain_val_fraction: float,
    seed: int,
    dataset_name: str = "cifar10",
) -> Dict[str, object]:
    spec = get_dataset_spec(dataset_name)
    invalid_classes = sorted(
        class_id
        for class_id in forget_classes
        if class_id < 0 or class_id >= len(spec.class_names)
    )
    if invalid_classes:
        raise ValueError(
            f"Invalid forget classes for {spec.name}: {invalid_classes}; "
            f"expected ids from 0 to {len(spec.class_names) - 1}"
        )

    ensure_dir(data_dir)
    ensure_dir(Path(split_path).parent)
    train_ds, test_ds = _load_dataset_pair(data_dir, spec.name, download=True)
    splits: Dict[str, object] = make_splits(
        train_ds.targets,
        test_ds.targets,
        SplitConfig(
            forget_classes=forget_classes,
            forget_fraction=forget_fraction,
            retain_val_fraction=retain_val_fraction,
            seed=seed,
        ),
    )
    splits["dataset"] = spec.name
    splits["class_names"] = list(spec.class_names)
    save_json(splits, split_path)
    return splits


def make_collate_fn(image_processor: CLIPImageProcessor):
    return CLIPCollator(image_processor=image_processor)


def build_text_inputs(
    tokenizer: CLIPTokenizer,
    class_names: Sequence[str] = CIFAR10_CLASSES,
    template: str = DEFAULT_PROMPT_TEMPLATE,
    max_length: int = 32,
) -> Dict[str, torch.Tensor]:
    prompts = [template.format(name) for name in class_names]
    tokens = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
    }


def build_loaders(
    data_dir: str,
    split_path: str,
    image_processor: CLIPImageProcessor,
    batch_size: int,
    num_workers: int,
    dataset_name: str | None = None,
) -> Dict[str, DataLoader]:
    split, spec, _ = load_split_metadata(split_path, dataset_name)
    train_ds, test_ds = _load_dataset_pair(data_dir, spec.name, download=False)
    collate_fn = make_collate_fn(image_processor)

    def _loader(
        dataset: Dataset, shuffle: bool, drop_last: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )

    subsets = {
        "forget": CIFARSubset(train_ds, split["forget_indices"]),
        "retain_train": CIFARSubset(train_ds, split["retain_train_indices"]),
        "retain_val": CIFARSubset(train_ds, split["retain_val_indices"]),
        "finetune_train": CIFARSubset(train_ds, split["finetune_train_indices"]),
        "test_forget": CIFARSubset(test_ds, split["test_forget_indices"]),
        "test_retain": CIFARSubset(test_ds, split["test_retain_indices"]),
        "test_all": CIFARSubset(test_ds, split["test_all_indices"]),
    }

    return {
        "forget": _loader(subsets["forget"], shuffle=True),
        "retain_train": _loader(subsets["retain_train"], shuffle=True),
        "retain_val": _loader(subsets["retain_val"], shuffle=False),
        "finetune_train": _loader(subsets["finetune_train"], shuffle=True),
        "test_forget": _loader(subsets["test_forget"], shuffle=False),
        "test_retain": _loader(subsets["test_retain"], shuffle=False),
        "test_all": _loader(subsets["test_all"], shuffle=False),
    }


def summarize_splits(split_path: str) -> Mapping[str, int]:
    split = load_json(split_path)
    return {
        "forget_count": len(split["forget_indices"]),
        "retain_train_count": len(split["retain_train_indices"]),
        "retain_val_count": len(split["retain_val_indices"]),
        "finetune_train_count": len(split["finetune_train_indices"]),
        "test_forget_count": len(split["test_forget_indices"]),
        "test_retain_count": len(split["test_retain_indices"]),
        "test_all_count": len(split["test_all_indices"]),
    }


def cycle_loader(loader: DataLoader) -> Iterable[Dict[str, torch.Tensor]]:
    while True:
        yield from loader
