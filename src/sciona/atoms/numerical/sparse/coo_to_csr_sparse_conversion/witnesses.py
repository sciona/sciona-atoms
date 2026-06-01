from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_validate_coo_dimensions(row: AbstractArray, col: AbstractArray, shape: AbstractScalar | int) -> AbstractScalar:
    """Ghost witness for validate_coo_dimensions."""
    _ = (row, col, shape)
    return AbstractScalar(dtype="float64")

def witness_lexsort_coo_arrays(row: AbstractArray, col: AbstractArray) -> AbstractArray:
    """Ghost witness for lexsort_coo_arrays."""
    _ = (row, col)
    return AbstractArray(shape=row.shape, dtype=row.dtype)

def witness_apply_sorting_and_consolidate(row: AbstractArray, col: AbstractArray, data: AbstractArray, permutation: AbstractArray) -> AbstractArray:
    """Ghost witness for apply_sorting_and_consolidate."""
    _ = (row, col, data, permutation)
    return AbstractArray(shape=row.shape, dtype=row.dtype)

def witness_build_csr_pointers(sorted_row: AbstractArray, sorted_col: AbstractArray, num_rows: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for build_csr_pointers."""
    _ = (sorted_row, sorted_col, num_rows)
    return AbstractArray(shape=sorted_row.shape, dtype=sorted_row.dtype)

