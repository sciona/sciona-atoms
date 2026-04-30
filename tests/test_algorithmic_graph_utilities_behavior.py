from __future__ import annotations

import numpy as np

from sciona.atoms.algorithmic.graph.utilities import (
    adjacency_to_edge_list,
    connected_components,
    edge_list_to_adjacency,
    graph_laplacian,
    knn_graph,
    molecular_distance_graph,
    node_degrees,
    pagerank,
    radius_graph,
    skeleton_to_graph,
)


def test_knn_graph_builds_fixed_out_degree_edges() -> None:
    points = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )

    edge_index, distances = knn_graph(points, k=1)

    assert edge_index.shape == (2, 4)
    assert distances.shape == (4,)
    assert np.allclose(distances, 1.0)
    assert np.array_equal(np.bincount(edge_index[0], minlength=4), np.ones(4, dtype=np.int64))


def test_radius_graph_connects_only_points_within_radius() -> None:
    points = np.asarray([[0.0], [1.0], [3.0]], dtype=np.float64)

    edge_index, distances = radius_graph(points, radius=1.5)

    assert {tuple(edge) for edge in edge_index.T.tolist()} == {(0, 1), (1, 0)}
    assert np.allclose(distances, 1.0)


def test_molecular_distance_graph_uses_cutoff_without_hydrogen_hydrogen_edge() -> None:
    coordinates = np.asarray(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]],
        dtype=np.float64,
    )
    elements = np.asarray(["O", "H", "H"])

    edge_index, distances, edge_features = molecular_distance_graph(
        coordinates,
        elements,
        cutoff=1.2,
    )

    assert {tuple(edge) for edge in edge_index.T.tolist()} == {(0, 1), (1, 0), (0, 2), (2, 0)}
    assert np.allclose(distances, 0.96)
    assert edge_features.shape == (4, 2)
    assert np.all(edge_features[:, 1] == 0.0)


def test_adjacency_edge_list_roundtrip_accumulates_parallel_edges() -> None:
    edge_index = np.asarray([[0, 0, 1], [1, 1, 0]], dtype=np.int64)

    adjacency = edge_list_to_adjacency(edge_index, num_nodes=2)
    rebuilt_edges = adjacency_to_edge_list(adjacency)

    assert np.array_equal(adjacency, np.asarray([[0.0, 2.0], [1.0, 0.0]]))
    assert {tuple(edge) for edge in rebuilt_edges.T.tolist()} == {(0, 1), (1, 0)}


def test_graph_laplacian_and_node_degrees() -> None:
    adjacency = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    edge_index = np.asarray([[0, 0, 1], [1, 2, 2]], dtype=np.int64)

    laplacian = graph_laplacian(adjacency)
    degrees = node_degrees(edge_index, num_nodes=3, mode="out")

    assert np.allclose(laplacian, np.asarray([[1.0, -1.0], [-1.0, 1.0]]))
    assert np.array_equal(degrees, np.asarray([2, 1, 0], dtype=np.int64))


def test_pagerank_center_node_highest_for_star_graph() -> None:
    adjacency = np.asarray(
        [
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    ranks = pagerank(adjacency)

    assert np.isclose(np.sum(ranks), 1.0)
    assert int(np.argmax(ranks)) == 0


def test_connected_components_labels_disjoint_pairs() -> None:
    adjacency = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    count, labels = connected_components(adjacency)

    assert count == 2
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_skeleton_to_graph_extracts_y_shape_endpoints_and_junction() -> None:
    skeleton = np.asarray(
        [
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.int64,
    )

    edge_index, edge_weights, node_coordinates = skeleton_to_graph(skeleton)

    assert node_coordinates.shape == (4, 2)
    assert edge_index.shape == (2, 3)
    assert edge_weights.shape == (3,)
    assert np.all(edge_weights > 0.0)

