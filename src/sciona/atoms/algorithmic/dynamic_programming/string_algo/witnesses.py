"""Ghost witnesses for dynamic programming string algorithms."""

from __future__ import annotations


def witness_init_edit_distance_table(source: str, target: str) -> list[list[int]]:
    """Ghost witness for init_edit_distance_table."""
    _ = (source, target)
    return [[]]


def witness_fill_edit_distance_table(
    table: list[list[int]],
    source: str,
    target: str,
) -> list[list[int]]:
    """Ghost witness for fill_edit_distance_table."""
    _ = (table, source, target)
    return [[]]
