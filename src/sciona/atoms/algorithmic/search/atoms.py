"""Search algorithm atoms."""

from __future__ import annotations

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_binary_search_midpoint,
    witness_binary_search_compare,
)


@register_atom(witness_binary_search_midpoint)
@icontract.require(lambda low: low >= 0, "Low bound must be non-negative")
@icontract.require(lambda high: high >= -1, "High bound must be valid")
@icontract.ensure(lambda result, low, high: low <= result <= max(low, high), "Midpoint must be within bounds")
def binary_search_midpoint(low: int, high: int) -> int:
    """Compute midpoint index for binary search bounds.

    Args:
        low: Lower index bound of the search range.
        high: Upper index bound of the search range.

    Returns:
        The midpoint index.
    """
    return low + (high - low) // 2


@register_atom(witness_binary_search_compare)
@icontract.require(lambda sorted_array: all(sorted_array[i] <= sorted_array[i + 1] for i in range(len(sorted_array) - 1)), "Array must be sorted")
@icontract.require(lambda sorted_array, mid: 0 <= mid < len(sorted_array) if sorted_array else mid >= 0, "Midpoint index must be valid")
@icontract.ensure(lambda result: result[0] >= 0 and result[1] >= -1, "Returned bounds must be valid indices")
def binary_search_compare(
    sorted_array: list[int],
    target: int,
    mid: int,
    low: int = 0,
    high: int = -1,
) -> tuple[int, int]:
    """Compare target value against midpoint element and narrow bounds.

    Args:
        sorted_array: The sorted list of integers to search.
        target: The target value to find.
        mid: The current midpoint index.
        low: The current lower bound.
        high: The current upper bound.

    Returns:
        A tuple of (low, high) representing the narrowed bounds.
    """
    if high == -1:
        high = len(sorted_array) - 1

    if not sorted_array:
        return 0, -1

    if sorted_array[mid] == target:
        return mid, mid
    elif sorted_array[mid] < target:
        return mid + 1, high
    else:
        return low, mid - 1
