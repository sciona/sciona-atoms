from __future__ import annotations

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_jaccard_similarity_threshold,
)


def _sets_valid(sets: list[set[int] | set[str]]) -> bool:
    if not isinstance(sets, list):
        return False
    for s in sets:
        if not isinstance(s, (set, frozenset)):
            return False
    return True


@register_atom(witness_jaccard_similarity_threshold, name="jaccard_similarity_threshold")
@icontract.require(lambda sets: _sets_valid(sets), "sets must be a list of sets")
@icontract.require(lambda threshold: isinstance(threshold, (int, float)) and 0.0 <= float(threshold) <= 1.0, "threshold must be in [0, 1]")
@icontract.ensure(
    lambda result, sets, threshold: all(
        isinstance(i, int) and isinstance(j, int) and isinstance(sim, float)
        and 0 <= i < j < len(sets)
        and sim >= float(threshold)
        for i, j, sim in result
    ),
    "returned pairs must be valid indices with similarity above the threshold"
)
def jaccard_similarity_threshold(
    sets: list[set[int] | set[str]],
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Compare multiple sets and return pairs with Jaccard similarity >= threshold.

    Args:
        sets: list of sets to compare.
        threshold: minimum similarity threshold in [0, 1].
    """
    results = []
    n = len(sets)
    threshold_val = float(threshold)
    for i in range(n):
        s_i = sets[i]
        len_i = len(s_i)
        for j in range(i + 1, n):
            s_j = sets[j]
            len_j = len(s_j)
            if len_i == 0 and len_j == 0:
                sim = 1.0
            else:
                intersection = len(s_i & s_j)
                union = len_i + len_j - intersection
                sim = float(intersection) / union if union > 0 else 0.0
            if sim >= threshold_val:
                results.append((i, j, sim))
    return results
