"""Dynamic programming string algorithms."""

from __future__ import annotations

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_init_edit_distance_table,
    witness_fill_edit_distance_table,
)


@register_atom(witness_init_edit_distance_table)
@icontract.require(lambda source, target: source is not None and target is not None, "Source and target strings must not be None")
@icontract.ensure(lambda result, source, target: len(result) == len(source) + 1 and all(len(row) == len(target) + 1 for row in result), "Initialized table must match dimensions of strings")
@icontract.ensure(lambda result: all(result[i][0] == i for i in range(len(result))), "Base cases for source deletions must be initialized")
def init_edit_distance_table(source: str, target: str) -> list[list[int]]:
    """Initialize the edit distance DP table with base cases.

    Args:
        source: The source string.
        target: The target string.

    Returns:
        An initialized DP table with base cases filled.
    """
    m = len(source)
    n = len(target)
    table = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        table[i][0] = i
    for j in range(n + 1):
        table[0][j] = j
    return table


@register_atom(witness_fill_edit_distance_table)
@icontract.require(lambda table, source, target: len(table) == len(source) + 1 and all(len(row) == len(target) + 1 for row in table), "Table must have matching dimensions")
@icontract.ensure(lambda result, table: len(result) == len(table) and all(len(row) == len(table[0]) for row in result), "Output dimensions must match input table")
def fill_edit_distance_table(
    table: list[list[int]],
    source: str,
    target: str,
) -> list[list[int]]:
    """Fill the edit distance table using recurrence relation.

    Args:
        table: The initialized/partially filled DP table.
        source: The source string.
        target: The target string.

    Returns:
        The fully filled edit distance DP table.
    """
    m = len(source)
    n = len(target)
    filled = [row.copy() for row in table]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if source[i - 1] == target[j - 1]:
                cost = 0
            else:
                cost = 1
            filled[i][j] = min(
                filled[i - 1][j] + 1,       # deletion
                filled[i][j - 1] + 1,       # insertion
                filled[i - 1][j - 1] + cost  # substitution
            )
    return filled
