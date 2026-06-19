"""Ghost witnesses for divide-and-conquer sorting atoms."""

from __future__ import annotations


def witness_split_list_halves(items: list[int]) -> tuple[list[int], list[int]]:
    """Ghost witness for split_list_halves."""
    _ = items
    return ([], [])


def witness_merge_sorted_halves(left: list[int], right: list[int]) -> list[int]:
    """Ghost witness for merge_sorted_halves."""
    _ = (left, right)
    return []
