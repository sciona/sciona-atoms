from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_gram_matrix(A: AbstractArray, b: AbstractArray) -> Tuple[AbstractArray, AbstractArray]:
    """Ghost witness for compute_gram_matrix."""
    _ = (A, b)
    n = A.shape[1]
    b_shape = b.shape
    if len(b_shape) > 1:
        ab_shape = (n, b_shape[1])
    else:
        ab_shape = (n,)
    return AbstractArray(shape=(n, n), dtype=A.dtype), AbstractArray(shape=ab_shape, dtype=A.dtype)

def witness_apply_tikhonov_shift_and_solve(Gram: AbstractArray, Ab: AbstractArray, alpha: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for apply_tikhonov_shift_and_solve."""
    _ = (Gram, Ab, alpha)
    return AbstractArray(shape=Ab.shape, dtype=Ab.dtype)


