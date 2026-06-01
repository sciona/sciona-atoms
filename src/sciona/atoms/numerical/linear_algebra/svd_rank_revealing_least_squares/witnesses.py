from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_svd_decompose(A: AbstractArray) -> AbstractArray:
    """Ghost witness for svd_decompose."""
    _ = (A)
    return AbstractArray(shape=A.shape, dtype=A.dtype)

def witness_svd_threshold_solve(U: AbstractArray, s: AbstractArray, Vh: AbstractArray, b: AbstractArray, cond: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for svd_threshold_solve."""
    _ = (U, s, Vh, b, cond)
    return AbstractArray(shape=U.shape, dtype=U.dtype)

