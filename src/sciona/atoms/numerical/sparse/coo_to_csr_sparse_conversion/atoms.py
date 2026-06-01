from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_validate_coo_dimensions,
    witness_lexsort_coo_arrays,
    witness_apply_sorting_and_consolidate,
    witness_build_csr_pointers,
)

@register_atom(witness_validate_coo_dimensions, name="validate_coo_dimensions")
@icontract.require(lambda row, col, shape: row.ndim == 1, "Precondition failed: row.ndim == 1")
@icontract.require(lambda row, col, shape: col.ndim == 1, "Precondition failed: col.ndim == 1")
@icontract.require(lambda row, col, shape: row.shape[0] == col.shape[0], "Precondition failed: row.shape[0] == col.shape[0]")
@icontract.ensure(lambda result, row, col, shape: result is not None, "Postcondition failed: result is not None")
def validate_coo_dimensions(row: NDArray[np.int64], col: NDArray[np.int64], shape: int) -> bool:
    """Verify row and column coordinate ranges against matrix shape bounds.

    Args:
        row: NDArray[np.int64]
        col: NDArray[np.int64]
        shape: tuple[int, int]

    Returns:
        is_valid: bool
    """
    import scipy.sparse._coo.coo_matrix
    return scipy.sparse._coo.coo_matrix._check_bounds(row=row, col=col, shape=shape) # type: ignore

@register_atom(witness_lexsort_coo_arrays, name="lexsort_coo_arrays")
@icontract.require(lambda row, col: row.shape[0] == col.shape[0], "Precondition failed: row.shape[0] == col.shape[0]")
@icontract.ensure(lambda result, row, col: permutation.shape[0] == row.shape[0], "Postcondition failed: permutation.shape[0] == row.shape[0]")
def lexsort_coo_arrays(row: NDArray[np.int64], col: NDArray[np.int64]) -> NDArray[np.int64]:
    """Obtain sorting permutation indices sorting primarily by row then by column.

    Args:
        row: NDArray[np.int64]
        col: NDArray[np.int64]

    Returns:
        permutation: NDArray[np.int64]
    """
    import numpy
    return numpy.lexsort(row=row, col=col) # type: ignore

@register_atom(witness_apply_sorting_and_consolidate, name="apply_sorting_and_consolidate")
@icontract.require(lambda row, col, data, permutation: row.shape[0] == permutation.shape[0], "Precondition failed: row.shape[0] == permutation.shape[0]")
@icontract.ensure(lambda result, row, col, data, permutation: sorted_row.shape[0] <= row.shape[0], "Postcondition failed: sorted_row.shape[0] <= row.shape[0]")
def apply_sorting_and_consolidate(row: NDArray[np.int64], col: NDArray[np.int64], data: NDArray[np.float64], permutation: NDArray[np.int64]) -> NDArray[np.int64]:
    """Sort data arrays and aggregate values falling on duplicate indices.

    Args:
        row: NDArray[np.int64]
        col: NDArray[np.int64]
        data: NDArray[np.float64]
        permutation: NDArray[np.int64]

    Returns:
        sorted_row: NDArray[np.int64]
    """
    import scipy.sparse._coo.coo_matrix
    return scipy.sparse._coo.coo_matrix.sum_duplicates(row=row, col=col, data=data, permutation=permutation) # type: ignore

@register_atom(witness_build_csr_pointers, name="build_csr_pointers")
@icontract.require(lambda sorted_row, sorted_col, num_rows: num_rows > 0, "Precondition failed: num_rows > 0")
@icontract.ensure(lambda result, sorted_row, sorted_col, num_rows: indptr.shape[0] == num_rows + 1, "Postcondition failed: indptr.shape[0] == num_rows + 1")
def build_csr_pointers(sorted_row: NDArray[np.int64], sorted_col: NDArray[np.int64], num_rows: int) -> NDArray[np.int64]:
    """Compress row indices into indptr arrays.

    Args:
        sorted_row: NDArray[np.int64]
        sorted_col: NDArray[np.int64]
        num_rows: int

    Returns:
        indptr: NDArray[np.int64]
    """
    import scipy.sparse._coo.coo_matrix
    return scipy.sparse._coo.coo_matrix._to_csr(sorted_row=sorted_row, sorted_col=sorted_col, num_rows=num_rows) # type: ignore

