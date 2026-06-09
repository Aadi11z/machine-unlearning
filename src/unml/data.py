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

CIFAR100_SUPERCLASS_NAMES = {
    "aquatic_mammals": ("beaver", "dolphin", "otter", "seal", "whale"),
    "fish": ("aquarium fish", "flatfish", "ray", "shark", "trout"),
    "flowers": ("orchid", "poppy", "rose", "sunflower", "tulip"),
    "food_containers": ("bottle", "bowl", "can", "cup", "plate"),
    "fruit_and_vegetables": (
        "apple",
        "mushroom",
        "orange",
        "pear",
        "sweet pepper",
    ),
    "household_electrical_devices": (
        "clock",
        "keyboard",
        "lamp",
        "telephone",
        "television",
    ),
    "household_furniture": ("bed", "chair", "couch", "table", "wardrobe"),
    "insects": ("bee", "beetle", "butterfly", "caterpillar", "cockroach"),
    "large_carnivores": ("bear", "leopard", "lion", "tiger", "wolf"),
    "large_man_made_outdoor_things": (
        "bridge",
        "castle",
        "house",
        "road",
        "skyscraper",
    ),
    "large_omnivores_and_herbivores": (
        "camel",
        "cattle",
        "chimpanzee",
        "elephant",
        "kangaroo",
    ),
    "medium_sized_mammals": (
        "fox",
        "porcupine",
        "possum",
        "raccoon",
        "skunk",
    ),
    "non_insect_invertebrates": ("crab", "lobster", "snail", "spider", "worm"),
    "people": ("baby", "boy", "girl", "man", "woman"),
    "reptiles": ("crocodile", "dinosaur", "lizard", "snake", "turtle"),
    "small_mammals": ("hamster", "mouse", "rabbit", "shrew", "squirrel"),
    "trees": (
        "maple tree",
        "oak tree",
        "palm tree",
        "pine tree",
        "willow tree",
    ),
    "vehicles_1": ("bicycle", "bus", "motorcycle", "pickup truck", "train"),
    "vehicles_2": ("lawn mower", "rocket", "streetcar", "tank", "tractor"),
    "natural_outdoor_scenes": ("cloud", "forest", "mountain", "plain", "sea"),
}

CIFAR100_SUPERCLASSES = {
    name: tuple(CIFAR100_CLASSES.index(class_name) for class_name in class_names)
    for name, class_names in CIFAR100_SUPERCLASS_NAMES.items()
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_class: type[Dataset]
    class_names: tuple[str, ...]
    superclasses: Mapping[str, tuple[int, ...]]


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec("cifar10", CIFAR10, CIFAR10_CLASSES, {}),
    "cifar100": DatasetSpec(
        "cifar100", CIFAR100, CIFAR100_CLASSES, CIFAR100_SUPERCLASSES
    ),
}


@dataclass
class SplitConfig:
    forget_classes: Sequence[int]
    forget_fraction: float
    retain_val_fraction: float
    seed: int
    request_name: str | None = None
    request_type: str | None = None
    superclass: str | None = None
    sibling_classes: Sequence[int] = ()
    unrelated_classes: Sequence[int] = ()


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


def validate_hierarchy_request(
    spec: DatasetSpec,
    cfg: SplitConfig,
) -> tuple[list[int], list[int]]:
    target_classes = sorted(set(int(class_id) for class_id in cfg.forget_classes))
    sibling_classes = sorted(set(int(class_id) for class_id in cfg.sibling_classes))
    all_classes = set(range(len(spec.class_names)))

    invalid_classes = sorted(
        class_id
        for class_id in [*target_classes, *sibling_classes]
        if class_id not in all_classes
    )
    if invalid_classes:
        raise ValueError(
            f"Invalid request classes for {spec.name}: {invalid_classes}; "
            f"expected ids from 0 to {len(spec.class_names) - 1}"
        )
    if set(target_classes) & set(sibling_classes):
        raise ValueError("Target and sibling classes must be disjoint")

    if cfg.request_name is None:
        return sibling_classes, sorted(
            all_classes - set(target_classes) - set(sibling_classes)
        )

    if spec.name != "cifar100":
        raise ValueError("Named hierarchy requests are currently supported for cifar100")
    if cfg.request_type not in {"superclass", "selective_class"}:
        raise ValueError(
            f"Unsupported request_type={cfg.request_type!r}; "
            "choose superclass or selective_class"
        )
    if cfg.superclass not in spec.superclasses:
        raise ValueError(
            f"Unknown CIFAR-100 superclass={cfg.superclass!r}; "
            f"choose from {sorted(spec.superclasses)}"
        )

    superclass_classes = set(spec.superclasses[cfg.superclass])
    if not set(target_classes).issubset(superclass_classes):
        raise ValueError("Target classes must belong to the configured superclass")
    if not set(sibling_classes).issubset(superclass_classes):
        raise ValueError("Sibling classes must belong to the configured superclass")

    if cfg.request_type == "superclass":
        if set(target_classes) != superclass_classes or sibling_classes:
            raise ValueError(
                "Superclass requests must target every class and have no siblings"
            )
    else:
        expected_siblings = superclass_classes - set(target_classes)
        if set(sibling_classes) != expected_siblings:
            raise ValueError(
                "Selective-class requests must list every non-target class "
                "in the superclass as a sibling"
            )

    unrelated_classes = sorted(all_classes - superclass_classes)
    return sibling_classes, unrelated_classes


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
    target_classes = sorted(set(int(class_id) for class_id in cfg.forget_classes))
    sibling_classes = sorted(set(int(class_id) for class_id in cfg.sibling_classes))
    excluded_hierarchy_classes = set(target_classes) | set(sibling_classes)
    unrelated_classes = sorted(set(int(value) for value in cfg.unrelated_classes))
    if not unrelated_classes:
        unrelated_classes = sorted(
            (set(train_labels) | set(test_labels)) - excluded_hierarchy_classes
        )

    rng = random.Random(cfg.seed)
    forget_pool = [
        i for i, label in enumerate(train_labels) if label in target_classes
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
    sibling_retain_train_indices = [
        i for i in retain_train_indices if train_labels[i] in sibling_classes
    ]
    unrelated_retain_train_indices = [
        i for i in retain_train_indices if train_labels[i] in unrelated_classes
    ]

    return {
        "forget_indices": forget_indices,
        "retain_train_indices": retain_train_indices,
        "retain_val_indices": retain_val_indices,
        "finetune_train_indices": sorted(retain_train_indices + forget_indices),
        "test_forget_indices": [
            i for i, label in enumerate(test_labels) if label in target_classes
        ],
        "test_retain_indices": [
            i for i, label in enumerate(test_labels) if label not in target_classes
        ],
        "sibling_retain_train_indices": sibling_retain_train_indices,
        "unrelated_retain_train_indices": unrelated_retain_train_indices,
        "test_sibling_indices": [
            i for i, label in enumerate(test_labels) if label in sibling_classes
        ],
        "test_unrelated_indices": [
            i for i, label in enumerate(test_labels) if label in unrelated_classes
        ],
        "test_all_indices": list(range(len(test_labels))),
        "forget_classes": target_classes,
        "target_classes": target_classes,
        "sibling_classes": sibling_classes,
        "unrelated_classes": unrelated_classes,
    }


def _load_dataset_pair(
    data_dir: str, dataset_name: str, download: bool
) -> tuple[Dataset, Dataset]:
    spec = get_dataset_spec(dataset_name)
    train_ds = spec.dataset_class(root=data_dir, train=True, download=download)
    test_ds = spec.dataset_class(root=data_dir, train=False, download=download)
    return train_ds, test_ds


def load_dataset_pair(
    data_dir: str, dataset_name: str, download: bool = False
) -> tuple[Dataset, Dataset]:
    """Load the configured train/test datasets for inspection or evaluation."""
    return _load_dataset_pair(data_dir, dataset_name, download)


def download_and_prepare_splits(
    data_dir: str,
    split_path: str,
    forget_classes: Sequence[int],
    forget_fraction: float,
    retain_val_fraction: float,
    seed: int,
    dataset_name: str = "cifar10",
    request_name: str | None = None,
    request_type: str | None = None,
    superclass: str | None = None,
    sibling_classes: Sequence[int] = (),
) -> Dict[str, object]:
    spec = get_dataset_spec(dataset_name)
    split_cfg = SplitConfig(
        forget_classes=forget_classes,
        forget_fraction=forget_fraction,
        retain_val_fraction=retain_val_fraction,
        seed=seed,
        request_name=request_name,
        request_type=request_type,
        superclass=superclass,
        sibling_classes=sibling_classes,
    )
    sibling_classes, unrelated_classes = validate_hierarchy_request(spec, split_cfg)
    split_cfg.sibling_classes = sibling_classes
    split_cfg.unrelated_classes = unrelated_classes

    ensure_dir(data_dir)
    ensure_dir(Path(split_path).parent)
    train_ds, test_ds = _load_dataset_pair(data_dir, spec.name, download=True)
    splits: Dict[str, object] = make_splits(
        train_ds.targets,
        test_ds.targets,
        split_cfg,
    )
    splits["dataset"] = spec.name
    splits["class_names"] = list(spec.class_names)
    splits["request_name"] = request_name
    splits["request_type"] = request_type
    splits["superclass"] = superclass
    splits["target_classes"] = sorted(set(int(value) for value in forget_classes))
    splits["sibling_classes"] = sibling_classes
    splits["unrelated_classes"] = unrelated_classes
    splits["seed"] = seed
    splits["forget_fraction"] = forget_fraction
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
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int | None = 2,
) -> Dict[str, DataLoader]:
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if prefetch_factor is not None and prefetch_factor < 1:
        raise ValueError("prefetch_factor must be at least 1")
    split, spec, _ = load_split_metadata(split_path, dataset_name)
    train_ds, test_ds = _load_dataset_pair(data_dir, spec.name, download=False)
    collate_fn = make_collate_fn(image_processor)

    def _loader(
        dataset: Dataset,
        shuffle: bool,
        drop_last: bool = False,
        reuse_workers: bool = False,
    ) -> DataLoader:
        worker_options = {}
        if num_workers > 0:
            worker_options = {
                "persistent_workers": persistent_workers and reuse_workers,
                "prefetch_factor": prefetch_factor,
            }
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=drop_last,
            **worker_options,
        )

    subsets = {
        "forget": CIFARSubset(train_ds, split["forget_indices"]),
        "retain_train": CIFARSubset(train_ds, split["retain_train_indices"]),
        "retain_val": CIFARSubset(train_ds, split["retain_val_indices"]),
        "finetune_train": CIFARSubset(train_ds, split["finetune_train_indices"]),
        "test_forget": CIFARSubset(test_ds, split["test_forget_indices"]),
        "test_retain": CIFARSubset(test_ds, split["test_retain_indices"]),
        "test_all": CIFARSubset(test_ds, split["test_all_indices"]),
        "sibling_retain": CIFARSubset(
            train_ds, split.get("sibling_retain_train_indices", [])
        ),
        "unrelated_retain": CIFARSubset(
            train_ds,
            split.get("unrelated_retain_train_indices", split["retain_train_indices"]),
        ),
        "test_sibling": CIFARSubset(
            test_ds, split.get("test_sibling_indices", [])
        ),
        "test_unrelated": CIFARSubset(
            test_ds, split.get("test_unrelated_indices", split["test_retain_indices"])
        ),
    }

    return {
        "forget": _loader(
            subsets["forget"], shuffle=True, reuse_workers=True
        ),
        "retain_train": _loader(
            subsets["retain_train"], shuffle=True, reuse_workers=True
        ),
        "retain_val": _loader(subsets["retain_val"], shuffle=False),
        "finetune_train": _loader(
            subsets["finetune_train"], shuffle=True, reuse_workers=True
        ),
        "test_forget": _loader(subsets["test_forget"], shuffle=False),
        "test_retain": _loader(subsets["test_retain"], shuffle=False),
        "test_all": _loader(subsets["test_all"], shuffle=False),
        "sibling_retain": _loader(
            subsets["sibling_retain"],
            shuffle=len(subsets["sibling_retain"]) > 0,
            reuse_workers=True,
        ),
        "unrelated_retain": _loader(
            subsets["unrelated_retain"],
            shuffle=len(subsets["unrelated_retain"]) > 0,
            reuse_workers=True,
        ),
        "test_sibling": _loader(subsets["test_sibling"], shuffle=False),
        "test_unrelated": _loader(subsets["test_unrelated"], shuffle=False),
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
        "sibling_retain_count": len(
            split.get("sibling_retain_train_indices", [])
        ),
        "unrelated_retain_count": len(
            split.get("unrelated_retain_train_indices", [])
        ),
        "test_sibling_count": len(split.get("test_sibling_indices", [])),
        "test_unrelated_count": len(split.get("test_unrelated_indices", [])),
    }


def cycle_loader(loader: DataLoader) -> Iterable[Dict[str, torch.Tensor]]:
    while True:
        yield from loader
