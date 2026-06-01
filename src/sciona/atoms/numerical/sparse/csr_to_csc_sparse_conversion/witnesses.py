from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_decompress_csr_pointers(indptr: AbstractArray, nnz: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for decompress_csr_pointers."""
    _ = (indptr, nnz)
    return AbstractArray(shape=indptr.shape, dtype=indptr.dtype)

def witness_transpose_indices_sort(row_coords: AbstractArray, col_indices: AbstractArray) -> AbstractArray:
    """Ghost witness for transpose_indices_sort."""
    _ = (row_coords, col_indices)
    return AbstractArray(shape=row_coords.shape, dtype=row_coords.dtype)

