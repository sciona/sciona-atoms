from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_decompress_csr_pointers,
    witness_transpose_indices_sort,
)

@register_atom(witness_decompress_csr_pointers, name="decompress_csr_pointers")
@icontract.require(lambda indptr, nnz: indptr.ndim == 1, "Precondition failed: indptr.ndim == 1")
@icontract.require(lambda indptr, nnz: indptr[-1] == nnz, "Precondition failed: indptr[-1] == nnz")
@icontract.ensure(lambda result, indptr, nnz: row_coordinates.shape[0] == nnz, "Postcondition failed: row_coordinates.shape[0] == nnz")
def decompress_csr_pointers(indptr: NDArray[np.int64], nnz: int) -> NDArray[np.int64]:
    """Convert row offset pointers back into raw row indices of length nnz.

    Args:
        indptr: NDArray[np.int64]
        nnz: int

    Returns:
        row_coordinates: NDArray[np.int64]
    """
    import scipy.sparse._sparsetools
    return scipy.sparse._sparsetools.expandptr(indptr=indptr, nnz=nnz) # type: ignore

@register_atom(witness_transpose_indices_sort, name="transpose_indices_sort")
@icontract.require(lambda row_coords, col_indices: row_coords.shape[0] == col_indices.shape[0], "Precondition failed: row_coords.shape[0] == col_indices.shape[0]")
@icontract.ensure(lambda result, row_coords, col_indices: result is not None, "Postcondition failed: result is not None")
def transpose_indices_sort(row_coords: NDArray[np.int64], col_indices: NDArray[np.int64]) -> NDArray[np.int64]:
    """Generate the sorted transpose indices map by column coordinates primarily.

    Args:
        row_coords: NDArray[np.int64]
        col_indices: NDArray[np.int64]

    Returns:
        transposed_perm: NDArray[np.int64]
    """
    import numpy
    return numpy.lexsort(row_coords=row_coords, col_indices=col_indices) # type: ignore

