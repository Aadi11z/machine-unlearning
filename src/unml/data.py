from __future__ import annotations
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Sequence, Mapping
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import InterpolationMode
from torchvision import transforms as T
from .utils import DEFAULT_PROMPT_TEMPLATE, ensure_dir, load_json, save_json

if TYPE_CHECKING:
    from transformers import CLIPImageProcessor, CLIPTokenizer

CIFAR10_CLASSES = [
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
]

@dataclass
class SplitConfig:
    forget_classes: Sequence[int]
    forget_fraction: float
    retain_val_fraction: float
    seed: int

class CIFARSubset(Dataset):
    def __init__(self, base: CIFAR10, indices: Sequence[int]):
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = self.indices[item]
        image, label = self.base[idx]
        return image, int(label), int(idx)

class CLIPCollator:
    # This portion is confusing!!! but here's the answer why it is needed:
    # Because of performance, control, and integration with DataLoader.
    # CLIPCollator:
        # Extracts config from image_processor
        # Uses torchvision transforms
        # Runs inside DataLoader workers
        # Avoids redundant conversions
    # So:
    # CLIPImageProcessor is used as a configuration source, not an execution engine.
  
    def __init__(self, image_processor):
        self.image_processor = image_processor
        # 224 — the shortest edge size used when resizing images (image_processor.size["shortest_edge"]).
        # {'height': 224, 'width': 224} — the final crop/output spatial size (images are produced as 224×224).
        # [0.48145466, 0.4578275, 0.40821073] — per-channel (R, G, B) mean used to normalize pixel values.
        # [0.26862954, 0.26130258, 0.27577711] — per-channel (R, G, B) std used to normalize pixel values.
        # Usage: the processor resizes/crops images to 224×224 and normalizes each pixel as 
        # (pixel - mean)/std (pixels expected as floats in [0,1]). These values come from the 
        # pretrained openai/clip-vit-base-patch32 preprocessing.
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

        mean = list(image_processor.image_mean) 
        std = list(image_processor.image_std)
        # the reason for these transforms are in the notes
        self.transform = T.Compose(
            [
                T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
                T.CenterCrop(crop_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, batch):
        images = [sample[0] for sample in batch]
        labels = torch.tensor([sample[1] for sample in batch], dtype=torch.long)
        indices = torch.tensor([sample[2] for sample in batch], dtype=torch.long)
        pixel_values = torch.stack([self.transform(image.convert("RGB")) for image in images], dim=0)
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "indices": indices,
        }

def _sample_indices(indices: List[int], fraction: float, rng: random.Random) -> List[int]:
    if fraction >= 0.999:
        return list(indices)
    count = max(1, int(len(indices) * fraction))
    return rng.sample(indices, count)

def make_splits(train_labels: Sequence[int], test_labels: Sequence[int], cfg: SplitConfig,) -> Dict[str, List[int]]:
    rng = random.Random(cfg.seed)
    forget_pool = [i for i, y in enumerate(train_labels) if y in cfg.forget_classes] # all indices whose labels belong to the forget labels
    forget_indices = sorted(_sample_indices(forget_pool, cfg.forget_fraction, rng))
    forget_set = set(forget_indices)    
    retain_all = [i for i in range(len(train_labels)) if i not in forget_set] # all indices whose labels belong to the retain set (!forget_set) 
    retain_val_size = max(1, int(len(retain_all) * cfg.retain_val_fraction))
    retain_val_indices = sorted(rng.sample(retain_all, retain_val_size))
    retain_val_set = set(retain_val_indices)
    retain_train_indices = sorted([i for i in retain_all if i not in retain_val_set])

    finetune_train_indices = sorted(retain_train_indices + forget_indices)

    test_forget_indices = [i for i, y in enumerate(test_labels) if y in cfg.forget_classes]
    test_retain_indices = [i for i, y in enumerate(test_labels) if y not in cfg.forget_classes]

    return {
        "forget_indices": forget_indices,
        "retain_train_indices": retain_train_indices,
        "retain_val_indices": retain_val_indices,
        "finetune_train_indices": finetune_train_indices,
        "test_forget_indices": test_forget_indices,
        "test_retain_indices": test_retain_indices,
        "test_all_indices": list(range(len(test_labels))),
        "forget_classes": list(cfg.forget_classes),
    }

def download_and_prepare_splits(data_dir: str, split_path: str, forget_classes: Sequence[int], forget_fraction: float, retain_val_fraction: float, seed: int) -> Dict[str, List[int]]:
    ensure_dir(data_dir)
    ensure_dir(Path(split_path).parent)
    train_ds = CIFAR10(root=data_dir, train=True, download=True)
    test_ds = CIFAR10(root=data_dir, train=False, download=True)

    split_cfg = SplitConfig(
        forget_classes=forget_classes,
        forget_fraction=forget_fraction,
        retain_val_fraction=retain_val_fraction,
        seed=seed,
    )
    splits = make_splits(train_ds.targets, test_ds.targets, split_cfg)
    save_json(splits, split_path)
    return splits

def load_cifar10(data_dir: str):
    train_ds = CIFAR10(root=data_dir, train=True, download=False)
    test_ds = CIFAR10(root=data_dir, train=False, download=False)
    return train_ds, test_ds

def make_collate_fn(image_processor: CLIPImageProcessor):
    return CLIPCollator(image_processor=image_processor)

def build_text_inputs(tokenizer: CLIPTokenizer, class_names: Sequence[str] = CIFAR10_CLASSES, template: str = DEFAULT_PROMPT_TEMPLATE, max_length: int = 32) -> Dict[str, torch.Tensor]:
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

def build_loaders(data_dir: str, split_path: str, image_processor: CLIPImageProcessor, batch_size: int, num_workers: int) -> Dict[str, DataLoader]:
    split = load_json(split_path)
    train_ds, test_ds = load_cifar10(data_dir)

    collate_fn = make_collate_fn(image_processor)

    def _loader(dataset: Dataset, shuffle: bool, drop_last: bool = False) -> DataLoader:
        # returns the dict by CLIPCollator
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
        "forget": _loader(subsets["forget"], shuffle=True, drop_last=False),
        "retain_train": _loader(subsets["retain_train"], shuffle=True, drop_last=False),
        "retain_val": _loader(subsets["retain_val"], shuffle=False),
        "finetune_train": _loader(subsets["finetune_train"], shuffle=True, drop_last=False),
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
        for batch in loader:
            yield batch
