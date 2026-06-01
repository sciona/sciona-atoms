from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_qr_factorize(A: AbstractArray, pivoting: AbstractScalar | bool) -> AbstractArray:
    """Ghost witness for qr_factorize."""
    _ = (A, pivoting)
    return AbstractArray(shape=A.shape, dtype=A.dtype)

def witness_qr_solve_least_squares(q: AbstractArray, r: AbstractArray, b: AbstractArray, p: AbstractArray) -> AbstractArray:
    """Ghost witness for qr_solve_least_squares."""
    _ = (q, r, b, p)
    return AbstractArray(shape=q.shape, dtype=q.dtype)

