from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_iterate_gmres(A: AbstractScalar | Any, b: AbstractArray, restart: AbstractScalar | int, tol: AbstractScalar | float, maxiter: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for iterate_gmres."""
    _ = (A, b, restart, tol, maxiter)
    return AbstractArray(shape=b.shape, dtype=b.dtype)

