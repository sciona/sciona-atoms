from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_check_and_factorize_metric(B: AbstractArray) -> AbstractArray:
    """Ghost witness for check_and_factorize_metric."""
    _ = (B)
    return AbstractArray(shape=B.shape, dtype=B.dtype)

def witness_solve_generalized_symmetric(A: AbstractArray, c_factor: AbstractArray) -> AbstractArray:
    """Ghost witness for solve_generalized_symmetric."""
    _ = (A, c_factor)
    return AbstractArray(shape=A.shape, dtype=A.dtype)

