from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_iterate_conjugate_gradient(A: AbstractScalar | Any, b: AbstractArray, M: AbstractScalar | Any, tol: AbstractScalar | float, maxiter: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for iterate_conjugate_gradient."""
    _ = (A, b, M, tol, maxiter)
    return AbstractArray(shape=b.shape, dtype=b.dtype)

