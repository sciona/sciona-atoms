"""Algorithmic search wrappers for the namespace pilot."""

from __future__ import annotations

import icontract
import numpy as np

SearchKey = int | float | str | bool | complex | bytes | np.generic


@icontract.require(lambda arr: len(arr) > 0, "Array must be non-empty")
@icontract.require(
    lambda arr: all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1)),
    "Array must be sorted for binary search",
)
@icontract.ensure(lambda result: result >= -1, "Result must be valid index or -1")
def binary_search(arr: np.ndarray, key: SearchKey) -> int:
    """Binary search over a sorted one-dimensional array."""
    idx = np.searchsorted(arr, key)
    if idx < len(arr) and arr[idx] == key:
        return int(idx)
    return -1


@icontract.require(lambda arr: len(arr) > 0, "Array must be non-empty")
@icontract.ensure(lambda result: result >= -1, "Result must be valid index or -1")
def linear_search(arr: np.ndarray, key: SearchKey) -> int:
    """Linear search over an arbitrary one-dimensional array."""
    matches = np.where(arr == key)[0]
    if len(matches) > 0:
        return int(matches[0])
    return -1


@icontract.require(lambda arr: len(arr) > 0, "Array must be non-empty")
@icontract.ensure(lambda result: result >= -1, "Result must be valid index or -1")
def hash_lookup(arr: np.ndarray, key: SearchKey) -> int:
    """Hash-backed lookup that returns the first matching index."""
    table: dict[object, int] = {}
    for i, value in enumerate(arr):
        item = value.item() if hasattr(value, "item") else value
        if item not in table:
            table[item] = i
    probe = key.item() if hasattr(key, "item") else key
    return table.get(probe, -1)
