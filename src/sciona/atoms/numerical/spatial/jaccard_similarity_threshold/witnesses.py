from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_jaccard_similarity_threshold(
    sets: list[set[int] | set[str]],
    threshold: float,
) -> AbstractArray:
    """Ghost witness for jaccard_similarity_threshold."""
    _ = (sets, threshold)
    n = len(sets) if sets else 0
    return AbstractArray(shape=(n * (n - 1) // 2 if n > 0 else 0, 3), dtype="object")
