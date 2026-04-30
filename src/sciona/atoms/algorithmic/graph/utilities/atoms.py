"""Generic graph construction, conversion, and feature atoms."""

from __future__ import annotations

import math

import icontract
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_adjacency_to_edge_list,
    witness_connected_components,
    witness_edge_list_to_adjacency,
    witness_graph_laplacian,
    witness_knn_graph,
    witness_molecular_distance_graph,
    witness_node_degrees,
    witness_pagerank,
    witness_radius_graph,
    witness_skeleton_to_graph,
)


def _square_matrix(adjacency: NDArray[np.float64]) -> bool:
    return adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1]


def _valid_edge_index(edge_index: NDArray[np.int64], num_nodes: int) -> bool:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or num_nodes < 0:
        return False
    if edge_index.shape[1] == 0:
        return True
    return bool(np.min(edge_index) >= 0 and np.max(edge_index) < num_nodes)


def _metric_to_p(metric: str) -> float:
    if metric == "euclidean":
        return 2.0
    if metric == "manhattan":
        return 1.0
    return math.inf


def _active_neighbors(point: tuple[int, int], active: NDArray[np.bool_]) -> list[tuple[int, int]]:
    row, col = point
    rows, cols = active.shape
    neighbors: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr = row + dr
            nc = col + dc
            if 0 <= nr < rows and 0 <= nc < cols and active[nr, nc]:
                neighbors.append((nr, nc))
    return neighbors


def _step_length(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


@register_atom(witness_knn_graph)
@icontract.require(lambda features: features.ndim == 2, "features must be a 2-D array")
@icontract.require(lambda features, k: 0 < k < features.shape[0], "k must be between 1 and n_nodes - 1")
@icontract.require(lambda metric: metric in {"euclidean", "manhattan", "chebyshev"}, "metric must be supported")
@icontract.ensure(lambda features, k, result: result[0].shape == (2, features.shape[0] * k), "edge index has one k-neighbor edge per source")
@icontract.ensure(lambda result: result[1].ndim == 1 and np.all(result[1] >= 0.0), "distances must be non-negative")
def knn_graph(
    features: NDArray[np.float64],
    k: int,
    metric: str = "euclidean",
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Build a directed k-nearest-neighbor graph from node feature rows."""
    from scipy.spatial import cKDTree

    n_nodes = features.shape[0]
    tree = cKDTree(features)
    distances, indices = tree.query(features, k=n_nodes, p=_metric_to_p(metric))

    sources = np.repeat(np.arange(n_nodes, dtype=np.int64), k)
    targets = np.empty(n_nodes * k, dtype=np.int64)
    edge_distances = np.empty(n_nodes * k, dtype=np.float64)
    for source in range(n_nodes):
        candidates = indices[source][indices[source] != source][:k]
        candidate_distances = distances[source][indices[source] != source][:k]
        start = source * k
        stop = start + k
        targets[start:stop] = candidates.astype(np.int64)
        edge_distances[start:stop] = candidate_distances.astype(np.float64)
    return np.vstack((sources, targets)), edge_distances


@register_atom(witness_radius_graph)
@icontract.require(lambda coordinates: coordinates.ndim == 2, "coordinates must be a 2-D array")
@icontract.require(lambda radius: radius > 0.0, "radius must be positive")
@icontract.require(lambda p: p >= 1.0, "Minkowski order p must be at least 1")
@icontract.ensure(lambda coordinates, result: result[0].shape[0] == 2 and _valid_edge_index(result[0], coordinates.shape[0]), "edge index must reference valid nodes")
@icontract.ensure(lambda radius, result: np.all(result[1] <= radius + 1e-12), "all returned distances must be within radius")
def radius_graph(
    coordinates: NDArray[np.float64],
    radius: float,
    p: float = 2.0,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Connect every ordered pair of distinct points within a distance radius."""
    from scipy.spatial import cKDTree

    tree = cKDTree(coordinates)
    neighbor_lists = tree.query_ball_point(coordinates, r=radius, p=p)
    sources: list[int] = []
    targets: list[int] = []
    distances: list[float] = []
    for source, neighbors in enumerate(neighbor_lists):
        for target in sorted(neighbors):
            if target == source:
                continue
            distance = float(np.linalg.norm(coordinates[source] - coordinates[target], ord=p))
            sources.append(source)
            targets.append(int(target))
            distances.append(distance)
    edge_index = np.asarray([sources, targets], dtype=np.int64)
    return edge_index, np.asarray(distances, dtype=np.float64)


@register_atom(witness_molecular_distance_graph)
@icontract.require(lambda coordinates: coordinates.ndim == 2 and coordinates.shape[1] == 3, "coordinates must have shape (n_atoms, 3)")
@icontract.require(lambda coordinates, elements: elements.ndim == 1 and elements.shape[0] == coordinates.shape[0], "elements must contain one label per atom")
@icontract.require(lambda cutoff: cutoff > 0.0, "cutoff must be positive")
@icontract.ensure(lambda coordinates, result: result[0].shape[0] == 2 and _valid_edge_index(result[0], coordinates.shape[0]), "edge index must reference valid atoms")
@icontract.ensure(lambda cutoff, result: np.all(result[1] < cutoff), "bond distances must be below the cutoff")
def molecular_distance_graph(
    coordinates: NDArray[np.float64],
    elements: NDArray[np.str_],
    cutoff: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Create a directed molecular proximity graph from 3-D atom coordinates."""
    from scipy.spatial.distance import cdist

    distances = cdist(coordinates, coordinates)
    mask = (distances < cutoff) & (~np.eye(coordinates.shape[0], dtype=bool))
    edge_index = np.vstack(np.nonzero(mask)).astype(np.int64)
    edge_distances = distances[edge_index[0], edge_index[1]].astype(np.float64)
    same_element = (elements[edge_index[0]] == elements[edge_index[1]]).astype(np.float64)
    edge_features = np.column_stack((edge_distances, same_element)).astype(np.float64)
    return edge_index, edge_distances, edge_features


@register_atom(witness_adjacency_to_edge_list)
@icontract.require(lambda adj_matrix: adj_matrix.ndim == 2 and adj_matrix.shape[0] == adj_matrix.shape[1], "adjacency must be square")
@icontract.ensure(lambda result: result.ndim == 2 and result.shape[0] == 2, "edge list must have shape (2, E)")
def adjacency_to_edge_list(adj_matrix: NDArray[np.float64]) -> NDArray[np.int64]:
    """Convert nonzero entries of a dense adjacency matrix into edge indices."""
    return np.vstack(np.nonzero(adj_matrix)).astype(np.int64)


@register_atom(witness_edge_list_to_adjacency)
@icontract.require(lambda edge_list: edge_list.ndim == 2 and edge_list.shape[0] == 2, "edge_list must have shape (2, E)")
@icontract.require(lambda edge_list, num_nodes: _valid_edge_index(edge_list, num_nodes), "edge indices must be valid for num_nodes")
@icontract.ensure(lambda num_nodes, result: result.shape == (num_nodes, num_nodes), "adjacency must be square with num_nodes rows")
def edge_list_to_adjacency(
    edge_list: NDArray[np.int64],
    num_nodes: int,
) -> NDArray[np.float64]:
    """Convert edge indices into a dense adjacency matrix with parallel-edge counts."""
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    if edge_list.shape[1] > 0:
        np.add.at(adjacency, (edge_list[0], edge_list[1]), 1.0)
    return adjacency


@register_atom(witness_graph_laplacian)
@icontract.require(_square_matrix, "adjacency must be a square 2-D matrix")
@icontract.require(lambda adjacency: np.allclose(adjacency, adjacency.T), "adjacency must be symmetric")
@icontract.ensure(lambda adjacency, result: result.shape == adjacency.shape, "Laplacian must preserve adjacency shape")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "Laplacian entries must be finite")
def graph_laplacian(
    adjacency: NDArray[np.float64],
    normalized: bool = True,
) -> NDArray[np.float64]:
    """Compute an unnormalized or symmetric normalized graph Laplacian."""
    degree = np.sum(adjacency, axis=1)
    if not normalized:
        return np.diag(degree) - adjacency
    inv_sqrt_degree = np.zeros_like(degree, dtype=np.float64)
    nonzero = degree > 0.0
    inv_sqrt_degree[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    scaled = inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    return np.diag(nonzero.astype(np.float64)) - scaled


@register_atom(witness_node_degrees)
@icontract.require(lambda edge_index: edge_index.ndim == 2 and edge_index.shape[0] == 2, "edge_index must have shape (2, E)")
@icontract.require(lambda edge_index, num_nodes: _valid_edge_index(edge_index, num_nodes), "edge indices must be valid for num_nodes")
@icontract.require(lambda mode: mode in {"out", "in", "total"}, "mode must be out, in, or total")
@icontract.ensure(lambda num_nodes, result: result.shape == (num_nodes,), "degree vector must have one value per node")
@icontract.ensure(lambda result: np.all(result >= 0), "degrees must be non-negative")
def node_degrees(
    edge_index: NDArray[np.int64],
    num_nodes: int,
    mode: str = "out",
) -> NDArray[np.int64]:
    """Compute in-degree, out-degree, or total directed degree from edge indices."""
    out_degree = np.bincount(edge_index[0], minlength=num_nodes).astype(np.int64)
    if mode == "out":
        return out_degree
    in_degree = np.bincount(edge_index[1], minlength=num_nodes).astype(np.int64)
    if mode == "in":
        return in_degree
    return out_degree + in_degree


@register_atom(witness_pagerank)
@icontract.require(_square_matrix, "adjacency must be a square 2-D matrix")
@icontract.require(lambda adjacency: adjacency.shape[0] > 0, "adjacency must contain at least one node")
@icontract.require(lambda damping: 0.0 < damping < 1.0, "damping must lie between 0 and 1")
@icontract.require(lambda max_iter: max_iter >= 1, "max_iter must be positive")
@icontract.require(lambda tol: tol > 0.0, "tol must be positive")
@icontract.ensure(lambda adjacency, result: result.shape == (adjacency.shape[0],), "PageRank must return one score per node")
@icontract.ensure(lambda result: np.isclose(np.sum(result), 1.0), "PageRank scores must sum to one")
def pagerank(
    adjacency: NDArray[np.float64],
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> NDArray[np.float64]:
    """Compute PageRank centrality scores by damped power iteration."""
    n_nodes = adjacency.shape[0]
    if n_nodes == 0:
        return np.asarray([], dtype=np.float64)
    matrix = adjacency.astype(np.float64, copy=True)
    column_sums = np.sum(matrix, axis=0)
    transition = np.empty_like(matrix, dtype=np.float64)
    sink = column_sums == 0.0
    transition[:, ~sink] = matrix[:, ~sink] / column_sums[~sink]
    transition[:, sink] = 1.0 / n_nodes
    ranks = np.full(n_nodes, 1.0 / n_nodes, dtype=np.float64)
    teleport = np.full(n_nodes, (1.0 - damping) / n_nodes, dtype=np.float64)
    for _ in range(max_iter):
        updated = teleport + damping * (transition @ ranks)
        if np.linalg.norm(updated - ranks, ord=1) < tol:
            ranks = updated
            break
        ranks = updated
    return ranks / np.sum(ranks)


@register_atom(witness_connected_components)
@icontract.require(lambda adj_matrix: adj_matrix.ndim == 2 and adj_matrix.shape[0] == adj_matrix.shape[1], "adjacency must be square")
@icontract.ensure(lambda adj_matrix, result: result[1].shape == (adj_matrix.shape[0],), "labels must contain one entry per node")
@icontract.ensure(lambda result: len(np.unique(result[1])) == result[0], "component count must match unique labels")
def connected_components(
    adj_matrix: NDArray[np.float64],
    directed: bool = False,
) -> tuple[int, NDArray[np.int64]]:
    """Label connected components in a dense adjacency matrix."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components as scipy_connected_components

    count, labels = scipy_connected_components(
        csr_matrix(adj_matrix),
        directed=directed,
        connection="weak",
        return_labels=True,
    )
    return int(count), labels.astype(np.int64)


@register_atom(witness_skeleton_to_graph)
@icontract.require(lambda skeleton: skeleton.ndim == 2, "skeleton must be a 2-D image")
@icontract.require(lambda skeleton: np.all((skeleton == 0) | (skeleton == 1)), "skeleton must be binary")
@icontract.ensure(lambda result: result[0].ndim == 2 and result[0].shape[0] == 2, "edge index must have shape (2, E)")
@icontract.ensure(lambda result: result[2].ndim == 2 and result[2].shape[1] == 2, "node coordinates must have row and column positions")
def skeleton_to_graph(
    skeleton: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]]:
    """Extract endpoint and junction topology from a 2-D binary skeleton."""
    active = skeleton.astype(bool)
    active_points = [tuple(point) for point in np.argwhere(active)]
    if not active_points:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
        )

    neighbor_counts = {point: len(_active_neighbors(point, active)) for point in active_points}
    node_points = sorted(point for point, count in neighbor_counts.items() if count != 2)
    if not node_points:
        node_points = [min(active_points)]
    node_ids = {point: idx for idx, point in enumerate(node_points)}
    node_set = set(node_points)

    visited_segments: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    edges: list[tuple[int, int]] = []
    weights: list[float] = []

    for start in node_points:
        for neighbor in _active_neighbors(start, active):
            first_segment = tuple(sorted((start, neighbor)))
            if first_segment in visited_segments:
                continue
            previous = start
            current = neighbor
            path_segments = [first_segment]
            length = _step_length(previous, current)
            while current not in node_set:
                choices = [point for point in _active_neighbors(current, active) if point != previous]
                if not choices:
                    break
                next_point = sorted(choices)[0]
                segment = tuple(sorted((current, next_point)))
                path_segments.append(segment)
                previous, current = current, next_point
                length += _step_length(previous, current)
            for segment in path_segments:
                visited_segments.add(segment)
            if current in node_set and current != start:
                edges.append((node_ids[start], node_ids[current]))
                weights.append(length)

    if edges:
        edge_index = np.asarray(edges, dtype=np.int64).T
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
    return (
        edge_index,
        np.asarray(weights, dtype=np.float64),
        np.asarray(node_points, dtype=np.int64),
    )
