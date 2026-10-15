"""Train/validation splits shared by the data loaders.

Each call uses its own seeded RNG, so splits are reproducible without
reseeding Python's global ``random`` module.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Hashable, Sequence


def random_split(n: int, val_split: float, seed: int = 42) -> tuple[list[int], list[int]]:
    """Shuffle 0..n-1 and hold out ``val_split`` of it (at least one row).

    Used for regression targets, where classes don't exist.
    """
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(n * val_split))
    return indices[n_val:], indices[:n_val]


def stratified_split(
    labels: Sequence[Hashable], val_split: float, seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split so every class keeps its share in both parts.

    Each class sends ``round(count * val_split)`` rows to validation but always
    keeps at least one in training, so a rare class can't vanish from the
    training set (which crashes XGBoost and misaligns ROC/probability
    columns). Validation always gets at least one row when there are two or
    more rows, even if that means taking a singleton class's only row.
    Returns ``(train_idx, val_idx)``.
    """
    rng = random.Random(seed)
    by_class: dict[Hashable, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_class[label].append(i)

    train: list[int] = []
    val: list[int] = []
    for label in sorted(by_class, key=str):
        idx = by_class[label]
        rng.shuffle(idx)
        n_val = min(int(round(len(idx) * val_split)), len(idx) - 1)
        val.extend(idx[:n_val])
        train.extend(idx[n_val:])

    if not val and len(train) > 1:
        # Tiny datasets: borrow one row from the largest class. If every class
        # has a single row this costs that class its only training row, but
        # without any validation row nothing can be evaluated or exported.
        largest = set(max(by_class.values(), key=len))
        moved = next(i for i in train if i in largest)
        train.remove(moved)
        val.append(moved)

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val
