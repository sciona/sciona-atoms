"""Graph shortest path atoms."""

from __future__ import annotations

from typing import TypeVar, Dict, Generic

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_initialize_distances,
    witness_relax_edges,
)

Node = TypeVar("Node")


class Graph(Generic[Node]):
    """A weighted directed graph representation using adjacency lists."""

    def __init__(self, adjacency: dict[Node, dict[Node, float]]) -> None:
        self.adjacency = adjacency

    def _get_nodes(self) -> set[Node]:
        """Return the set of all nodes in the graph."""
        nodes = set(self.adjacency.keys())
        for targets in self.adjacency.values():
            nodes.update(targets.keys())
        return nodes


@register_atom(witness_initialize_distances)
@icontract.require(lambda graph, source: source in graph._get_nodes(), "Source node must exist in the graph")
@icontract.ensure(lambda result, source: result[source] == 0.0, "Source distance must be zero")
@icontract.ensure(lambda result: all(d >= 0.0 or d == float("inf") for d in result.values()), "Distances must be non-negative or infinity")
def initialize_distances(graph: Graph[Node], source: Node) -> dict[Node, float]:
    """Initialize distance map for a weighted graph shortest path routine.

    Args:
        graph: Input weighted graph.
        source: Starting node.

    Returns:
        Initial distance dictionary mapping nodes to initial distance values.
    """
    distances: dict[Node, float] = {}
    for node in graph._get_nodes():
        distances[node] = float("inf")
    if source in distances:
        distances[source] = 0.0
    return distances


@register_atom(witness_relax_edges)
@icontract.require(lambda graph, distances: all(node in distances for node in graph._get_nodes()), "Distances must cover all graph nodes")
@icontract.require(lambda graph: all(weight >= 0 for targets in graph.adjacency.values() for weight in targets.values()), "Graph weights must be non-negative")
@icontract.ensure(lambda result, distances: all(result[node] <= distances[node] for node in distances), "Relaxation should not increase distances")
def relax_edges(graph: Graph[Node], distances: dict[Node, float]) -> dict[Node, float]:
    """Relax weighted edges to improve tentative shortest path distances.

    Args:
        graph: Input weighted graph.
        distances: Current tentative shortest path distances.

    Returns:
        Updated shortest path distances after one relaxation pass over all edges.
    """
    updated = distances.copy()
    for u in graph.adjacency:
        for v, weight in graph.adjacency[u].items():
            if updated[u] + weight < updated[v]:
                updated[v] = updated[u] + weight
    return updated
