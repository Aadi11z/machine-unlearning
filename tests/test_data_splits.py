from __future__ import annotations

from unml.data import SplitConfig, make_splits

def test_make_splits_has_disjoint_and_complete_partitions() -> None:
    train_labels = [0, 1, 1, 2, 1, 3, 4, 1, 5, 1]
    test_labels = [1, 0, 2, 1, 3, 4, 1, 5]

    cfg = SplitConfig(
        forget_classes=[1],
        forget_fraction=0.4,
        retain_val_fraction=0.25,
        seed=7,
    )
    splits = make_splits(train_labels=train_labels, test_labels=test_labels, cfg=cfg)

    forget = set(splits["forget_indices"])
    retain_train = set(splits["retain_train_indices"])
    retain_val = set(splits["retain_val_indices"])

    # Forget samples must come from the forget class pool only.
    assert all(train_labels[i] == 1 for i in forget)

    # Retain partitions must be disjoint from forget and from each other.
    assert forget.isdisjoint(retain_train)
    assert forget.isdisjoint(retain_val)
    assert retain_train.isdisjoint(retain_val)

    non_forget = set(range(len(train_labels))) - forget
    assert retain_train | retain_val == non_forget

    finetune_expected = sorted(list(forget | retain_train))
    assert splits["finetune_train_indices"] == finetune_expected

    assert splits["test_all_indices"] == list(range(len(test_labels)))
    assert all(test_labels[i] == 1 for i in splits["test_forget_indices"])
    assert all(test_labels[i] != 1 for i in splits["test_retain_indices"])


def test_make_splits_is_deterministic_for_same_seed() -> None:
    train_labels = [0, 1, 1, 2, 1, 3, 4, 1, 5, 1]
    test_labels = [1, 0, 2, 1, 3, 4, 1, 5]

    cfg_a = SplitConfig(forget_classes=[1], forget_fraction=0.5, retain_val_fraction=0.2, seed=99)
    cfg_b = SplitConfig(forget_classes=[1], forget_fraction=0.5, retain_val_fraction=0.2, seed=99)

    splits_a = make_splits(train_labels=train_labels, test_labels=test_labels, cfg=cfg_a)
    splits_b = make_splits(train_labels=train_labels, test_labels=test_labels, cfg=cfg_b)

    assert splits_a == splits_b
