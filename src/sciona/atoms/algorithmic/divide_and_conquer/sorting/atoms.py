"""Divide-and-conquer sorting atoms."""

from __future__ import annotations

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_split_list_halves,
    witness_merge_sorted_halves,
)


@register_atom(witness_split_list_halves)
@icontract.require(lambda items: len(items) >= 2, "List must have at least 2 elements to be split")
@icontract.ensure(lambda result, items: len(result[0]) + len(result[1]) == len(items), "Sum of half lengths must equal original")
def split_list_halves(items: list[int]) -> tuple[list[int], list[int]]:
    """Split a list into left and right halves.

    Args:
        items: List of integers to split.

    Returns:
        A tuple of two lists representing the left and right halves.
    """
    mid = len(items) // 2
    return items[:mid], items[mid:]


@register_atom(witness_merge_sorted_halves)
@icontract.require(lambda left: all(left[i] <= left[i + 1] for i in range(len(left) - 1)), "Left half must be sorted")
@icontract.require(lambda right: all(right[i] <= right[i + 1] for i in range(len(right) - 1)), "Right half must be sorted")
@icontract.ensure(lambda result, left, right: len(result) == len(left) + len(right), "Result length must be sum of input lengths")
@icontract.ensure(lambda result: all(result[i] <= result[i + 1] for i in range(len(result) - 1)), "Result must be sorted")
def merge_sorted_halves(left: list[int], right: list[int]) -> list[int]:
    """Merge two sorted halves into one sorted list.

    Args:
        left: Left sorted list of integers.
        right: Right sorted list of integers.

    Returns:
        A single merged and sorted list of integers.
    """
    merged: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged
