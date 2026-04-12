"""Graph shortest-path wrappers for the namespace pilot."""

from __future__ import annotations

import icontract
import numpy as np


@icontract.require(
    lambda adj: adj.ndim == 2 and adj.shape[0] == adj.shape[1],
    "Adjacency matrix must be square",
)
@icontract.require(
    lambda adj: np.all(adj >= 0),
    "Dijkstra requires non-negative weights",
)
@icontract.ensure(lambda result: np.all(result >= 0), "Distances must be non-negative")
def dijkstra(adj: np.ndarray, source: int = 0) -> np.ndarray:
    """Single-source shortest paths via Dijkstra."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra

    graph = csr_matrix(adj)
    return sp_dijkstra(graph, indices=source, directed=True)


@icontract.require(
    lambda adj: adj.ndim == 2 and adj.shape[0] == adj.shape[1],
    "Adjacency matrix must be square",
)
@icontract.ensure(
    lambda result, adj: result.shape == (adj.shape[0],),
    "Bellman-Ford result must have one entry per node",
)
@icontract.ensure(lambda result: result.ndim == 1, "Bellman-Ford result must be 1D")
def bellman_ford(adj: np.ndarray, source: int = 0) -> np.ndarray:
    """Single-source shortest paths via Bellman-Ford."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import bellman_ford as sp_bellman_ford

    graph = csr_matrix(adj)
    return sp_bellman_ford(graph, indices=source, directed=True)


@icontract.require(
    lambda adj: adj.ndim == 2 and adj.shape[0] == adj.shape[1],
    "Adjacency matrix must be square",
)
@icontract.ensure(
    lambda result, adj: result.shape == (adj.shape[0], adj.shape[1]),
    "Floyd-Warshall result must be square with same dimensions as input",
)
@icontract.ensure(lambda result: result.ndim == 2, "Floyd-Warshall result must be 2D")
def floyd_warshall(adj: np.ndarray) -> np.ndarray:
    """All-pairs shortest paths via Floyd-Warshall."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import floyd_warshall as sp_floyd_warshall

    graph = csr_matrix(adj)
    return sp_floyd_warshall(graph, directed=True)
