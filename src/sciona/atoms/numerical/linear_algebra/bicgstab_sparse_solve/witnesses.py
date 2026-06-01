from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_iterate_bicgstab(A: AbstractScalar | Any, b: AbstractArray, tol: AbstractScalar | float, maxiter: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for iterate_bicgstab."""
    _ = (A, b, tol, maxiter)
    return AbstractArray(shape=b.shape, dtype=b.dtype)

