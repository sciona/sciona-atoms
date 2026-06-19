"""Ghost witnesses for graph shortest path atoms."""

from __future__ import annotations

from typing import Any


def witness_initialize_distances(graph: Any, source: Any) -> dict[Any, float]:
    """Ghost witness for initialize_distances."""
    _ = (graph, source)
    return {}


def witness_relax_edges(graph: Any, distances: dict[Any, float]) -> dict[Any, float]:
    """Ghost witness for relax_edges."""
    _ = (graph, distances)
    return {}
