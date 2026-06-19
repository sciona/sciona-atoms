"""Ghost witnesses for search atoms."""

from __future__ import annotations


def witness_binary_search_midpoint(low: int, high: int) -> int:
    """Ghost witness for binary_search_midpoint."""
    _ = (low, high)
    return 0


def witness_binary_search_compare(
    sorted_array: list[int],
    target: int,
    mid: int,
    low: int = 0,
    high: int = -1,
) -> tuple[int, int]:
    """Ghost witness for binary_search_compare."""
    _ = (sorted_array, target, mid, low, high)
    return 0, 0
