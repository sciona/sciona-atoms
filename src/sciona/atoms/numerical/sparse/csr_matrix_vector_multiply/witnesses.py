from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_validate_spmv_shapes(indptr: AbstractArray, x: AbstractArray) -> AbstractScalar:
    """Ghost witness for validate_spmv_shapes."""
    _ = (indptr, x)
    return AbstractScalar(dtype="float64")

def witness_compute_spmv_kernel(indptr: AbstractArray, indices: AbstractArray, data: AbstractArray, x: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_spmv_kernel."""
    _ = (indptr, indices, data, x)
    return AbstractArray(shape=indptr.shape, dtype=indptr.dtype)

