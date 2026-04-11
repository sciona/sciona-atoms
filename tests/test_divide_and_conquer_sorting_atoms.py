from __future__ import annotations

import numpy as np

from sciona.atoms.algorithmic.divide_and_conquer.sorting import (
    counting_sort,
    merge_sort,
    quicksort,
    radix_sort,
)


def test_merge_sort_returns_sorted_copy() -> None:
    data = np.array([5, 1, 4, 2, 3])
    assert np.array_equal(merge_sort(data), np.array([1, 2, 3, 4, 5]))


def test_quicksort_returns_sorted_copy() -> None:
    data = np.array([9, 3, 7, 1])
    assert np.array_equal(quicksort(data), np.array([1, 3, 7, 9]))


def test_counting_and_radix_sort_handle_integer_inputs() -> None:
    data = np.array([12, 4, 7, 4, 0, 3], dtype=int)
    expected = np.array([0, 3, 4, 4, 7, 12], dtype=int)
    assert np.array_equal(counting_sort(data), expected)
    assert np.array_equal(radix_sort(data), expected)
