from __future__ import annotations

import numpy as np
import pytest
from icontract.errors import ViolationError

from sciona.atoms.algorithmic.divide_and_conquer.sorting import (
    split_list_halves,
    merge_sorted_halves,
)
from sciona.atoms.algorithmic.graph.shortest_paths import (
    Graph,
    initialize_distances,
    relax_edges,
)
from sciona.atoms.algorithmic.search import (
    binary_search_midpoint,
    binary_search_compare,
)
from sciona.atoms.algorithmic.divide_and_conquer.matrix import (
    split_matrix_quadrants,
    combine_matrix_quadrants,
)
from sciona.atoms.algorithmic.dynamic_programming.string_algo import (
    init_edit_distance_table,
    fill_edit_distance_table,
)


def test_merge_sort_atoms() -> None:
    # Test split
    left, right = split_list_halves([4, 2, 5, 1])
    assert left == [4, 2]
    assert right == [5, 1]

    # Precondition check: list size >= 2
    with pytest.raises(ViolationError):
        split_list_halves([1])

    # Test merge
    merged = merge_sorted_halves([2, 4], [1, 5])
    assert merged == [1, 2, 4, 5]

    # Precondition check: input lists must be sorted
    with pytest.raises(ViolationError):
        merge_sorted_halves([4, 2], [1, 5])


def test_shortest_path_atoms() -> None:
    adj = {
        "A": {"B": 1.0, "C": 4.0},
        "B": {"C": 2.0, "D": 5.0},
        "C": {"D": 1.0},
        "D": {},
    }
    graph = Graph(adj)

    # Initialize
    distances = initialize_distances(graph, "A")
    assert distances == {"A": 0.0, "B": float("inf"), "C": float("inf"), "D": float("inf")}

    # Precondition check: source must exist
    with pytest.raises(ViolationError):
        initialize_distances(graph, "Z")

    # Relax once
    updated = relax_edges(graph, distances)
    assert updated["B"] == 1.0
    assert updated["C"] == 3.0
    assert updated["D"] == 4.0


def test_binary_search_atoms() -> None:
    # Test midpoint
    assert binary_search_midpoint(0, 4) == 2
    assert binary_search_midpoint(0, 5) == 2
    assert binary_search_midpoint(3, 7) == 5

    # Precondition checks
    with pytest.raises(ViolationError):
        binary_search_midpoint(-1, 5)

    # Test compare
    arr = [1, 3, 5, 7, 9]
    # target is midpoint
    assert binary_search_compare(arr, 5, 2, 0, 4) == (2, 2)
    # target in right half
    assert binary_search_compare(arr, 7, 2, 0, 4) == (3, 4)
    # target in left half
    assert binary_search_compare(arr, 3, 2, 0, 4) == (0, 1)

    # Precondition checks
    with pytest.raises(ViolationError):
        binary_search_compare([5, 3], 3, 0)  # not sorted


def test_strassen_atoms() -> None:
    mat = np.array([[1, 2], [3, 4]])
    q11, q12, q21, q22 = split_matrix_quadrants(mat)
    np.testing.assert_equal(q11, [[1]])
    np.testing.assert_equal(q12, [[2]])
    np.testing.assert_equal(q21, [[3]])
    np.testing.assert_equal(q22, [[4]])

    # Precondition checks: dimensions must be even
    with pytest.raises(ViolationError):
        split_matrix_quadrants(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    # Combine
    combined = combine_matrix_quadrants(q11, q12, q21, q22)
    np.testing.assert_equal(combined, mat)

    # Precondition checks: shapes must match
    with pytest.raises(ViolationError):
        combine_matrix_quadrants(q11, np.array([[1, 2]]), q21, q22)


def test_edit_distance_atoms() -> None:
    source = "kitten"
    target = "sitting"

    # Initialize
    table = init_edit_distance_table(source, target)
    assert len(table) == 7
    assert len(table[0]) == 8
    assert table[3][0] == 3
    assert table[0][5] == 5

    # Fill
    filled = fill_edit_distance_table(table, source, target)
    assert filled[6][7] == 3  # min edit distance between kitten and sitting

    # Precondition checks
    with pytest.raises(ViolationError):
        fill_edit_distance_table([[0, 0]], source, target)
