from __future__ import annotations

import numpy as np

from sciona.atoms.numpy import search_sort as atoms


def test_binary_search_insertion_forwards_searchsorted_with_sorter() -> None:
    a = np.array([40, 10, 20, 30])
    sorter = np.argsort(a)
    values = np.array([5, 25, 40])

    actual = atoms.binary_search_insertion(a, values, side="right", sorter=sorter)
    expected = np.searchsorted(a, values, side="right", sorter=sorter)

    np.testing.assert_array_equal(actual, expected)


def test_lexicographic_indirect_sort_forwards_axis() -> None:
    keys = np.array(
        [
            [[2, 1, 2], [0, 0, 1]],
            [[3, 1, 2], [1, 0, 1]],
        ]
    )

    actual = atoms.lexicographic_indirect_sort(keys, axis=1)
    expected = np.lexsort(keys, axis=1)

    np.testing.assert_array_equal(actual, expected)


def test_partial_sort_partition_returns_partition_and_argpartition() -> None:
    a = np.array([[9, 1, 5, 3], [8, 2, 6, 4]])

    actual_partition, actual_argpartition = atoms.partial_sort_partition(a, kth=2, axis=1)

    np.testing.assert_array_equal(actual_partition, np.partition(a, kth=2, axis=1))
    np.testing.assert_array_equal(actual_argpartition, np.argpartition(a, kth=2, axis=1))


def test_partial_sort_partition_axis_none_matches_numpy_flattened_output() -> None:
    a = np.array([[9, 1, 5], [8, 2, 6]])

    actual_partition, actual_argpartition = atoms.partial_sort_partition(a, kth=3, axis=None)

    np.testing.assert_array_equal(actual_partition, np.partition(a, kth=3, axis=None))
    np.testing.assert_array_equal(actual_argpartition, np.argpartition(a, kth=3, axis=None))
