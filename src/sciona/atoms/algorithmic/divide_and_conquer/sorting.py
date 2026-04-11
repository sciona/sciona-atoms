"""Divide-and-conquer sorting reference wrappers for the namespace pilot."""

from __future__ import annotations

import icontract
import numpy as np


@icontract.require(lambda a: len(a) > 0, "Input array must be non-empty")
@icontract.ensure(lambda result, a: len(result) == len(a), "Output must have same length as input")
@icontract.ensure(lambda result: all(result[i] <= result[i + 1] for i in range(len(result) - 1)), "Output must be sorted")
def merge_sort(a: np.ndarray) -> np.ndarray:
    """Stable divide-and-conquer merge sort over a one-dimensional array."""
    return np.sort(a, kind="mergesort")


@icontract.require(lambda a: len(a) > 0, "Input array must be non-empty")
@icontract.ensure(lambda result, a: len(result) == len(a), "Output must have same length as input")
@icontract.ensure(lambda result: all(result[i] <= result[i + 1] for i in range(len(result) - 1)), "Output must be sorted")
def quicksort(a: np.ndarray) -> np.ndarray:
    """Average-case divide-and-conquer quicksort over a one-dimensional array."""
    return np.sort(a, kind="quicksort")


@icontract.require(lambda a: len(a) > 0, "Input array must be non-empty")
@icontract.ensure(lambda result, a: len(result) == len(a), "Output must have same length as input")
@icontract.ensure(lambda result: all(result[i] <= result[i + 1] for i in range(len(result) - 1)), "Output must be sorted")
def heapsort(a: np.ndarray) -> np.ndarray:
    """Heap-based comparison sort kept here as a family variant hint."""
    return np.sort(a, kind="heapsort")


@icontract.require(lambda a: len(a) > 0, "Input array must be non-empty")
@icontract.require(lambda a: np.issubdtype(a.dtype, np.integer), "Requires integer array")
@icontract.ensure(lambda result, a: len(result) == len(a), "Output must have same length as input")
@icontract.ensure(lambda result: all(result[i] <= result[i + 1] for i in range(len(result) - 1)), "Output must be sorted")
def counting_sort(a: np.ndarray) -> np.ndarray:
    """Count-based integer sorting for bounded discrete values."""
    if len(a) == 0:
        return a.copy()
    min_val = int(a.min())
    max_val = int(a.max())
    count = np.zeros(max_val - min_val + 1, dtype=np.intp)
    for v in a:
        count[int(v) - min_val] += 1
    result = np.empty_like(a)
    idx = 0
    for val in range(min_val, max_val + 1):
        c = count[val - min_val]
        result[idx : idx + c] = val
        idx += c
    return result


@icontract.require(lambda a: len(a) > 0, "Input array must be non-empty")
@icontract.require(lambda a: np.issubdtype(a.dtype, np.integer), "Requires integer array")
@icontract.require(lambda a: np.all(a >= 0), "Requires non-negative integers")
@icontract.ensure(lambda result, a: len(result) == len(a), "Output must have same length as input")
@icontract.ensure(lambda result: all(result[i] <= result[i + 1] for i in range(len(result) - 1)), "Output must be sorted")
def radix_sort(a: np.ndarray) -> np.ndarray:
    """Digit-wise integer sorting for non-negative values."""
    if len(a) == 0:
        return a.copy()
    result = a.copy()
    max_val = int(result.max())
    exp = 1
    while max_val // exp > 0:
        output = np.empty_like(result)
        count = np.zeros(10, dtype=np.intp)
        for v in result:
            digit = (int(v) // exp) % 10
            count[digit] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]
        for v in reversed(result):
            digit = (int(v) // exp) % 10
            count[digit] -= 1
            output[count[digit]] = v
        result = output
        exp *= 10
    return result
