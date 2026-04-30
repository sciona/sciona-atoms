"""Ghost witnesses for generic graph utility atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _rows(x: AbstractArray) -> int:
    first = x.shape[0] if x.shape else 1
    return first if isinstance(first, int) else 1


def witness_knn_graph(
    features: AbstractArray,
    k: int,
    metric: str = "euclidean",
) -> tuple[AbstractArray, AbstractArray]:
    """Return symbolic edge and distance arrays for a fixed-degree graph."""
    edges = max(_rows(features) * k, 0)
    return (
        AbstractArray(shape=(2, edges), dtype="int64", min_val=0.0),
        AbstractArray(shape=(edges,), dtype="float64", min_val=0.0),
    )


def witness_radius_graph(
    coordinates: AbstractArray,
    radius: float,
    p: float = 2.0,
) -> tuple[AbstractArray, AbstractArray]:
    """Return symbolic edge and distance arrays for a radius graph."""
    return (
        AbstractArray(shape=(2, "E"), dtype="int64", min_val=0.0),
        AbstractArray(shape=("E",), dtype="float64", min_val=0.0),
    )


def witness_molecular_distance_graph(
    coordinates: AbstractArray,
    elements: AbstractArray,
    cutoff: float,
) -> tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Return symbolic molecular edge topology and edge features."""
    return (
        AbstractArray(shape=(2, "E"), dtype="int64", min_val=0.0),
        AbstractArray(shape=("E",), dtype="float64", min_val=0.0),
        AbstractArray(shape=("E", 2), dtype="float64"),
    )


def witness_adjacency_to_edge_list(adj_matrix: AbstractArray) -> AbstractArray:
    """Return symbolic coordinate-form edges from a square matrix."""
    return AbstractArray(shape=(2, "E"), dtype="int64", min_val=0.0)


def witness_edge_list_to_adjacency(
    edge_list: AbstractArray,
    num_nodes: int,
) -> AbstractArray:
    """Return symbolic dense adjacency from coordinate-form edges."""
    return AbstractArray(shape=(num_nodes, num_nodes), dtype="float64", min_val=0.0)


def witness_graph_laplacian(
    adjacency: AbstractArray,
    normalized: bool = True,
) -> AbstractArray:
    """Return symbolic graph Laplacian with the input matrix shape."""
    return AbstractArray(shape=adjacency.shape, dtype="float64")


def witness_node_degrees(
    edge_index: AbstractArray,
    num_nodes: int,
    mode: str = "out",
) -> AbstractArray:
    """Return symbolic node-degree vector."""
    return AbstractArray(shape=(num_nodes,), dtype="int64", min_val=0.0)


def witness_pagerank(
    adjacency: AbstractArray,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> AbstractArray:
    """Return symbolic PageRank score vector."""
    n_nodes = _rows(adjacency)
    return AbstractArray(shape=(n_nodes,), dtype="float64", min_val=0.0, max_val=1.0)


def witness_connected_components(
    adj_matrix: AbstractArray,
    directed: bool = False,
) -> tuple[int, AbstractArray]:
    """Return symbolic component count and labels."""
    return 1, AbstractArray(shape=(_rows(adj_matrix),), dtype="int64", min_val=0.0)


def witness_skeleton_to_graph(
    skeleton: AbstractArray,
) -> tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Return symbolic topology extracted from a 2-D skeleton image."""
    return (
        AbstractArray(shape=(2, "E"), dtype="int64", min_val=0.0),
        AbstractArray(shape=("E",), dtype="float64", min_val=0.0),
        AbstractArray(shape=("N", 2), dtype="int64", min_val=0.0),
    )

