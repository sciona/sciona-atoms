from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_solve_nnls(A: AbstractArray, b: AbstractArray) -> AbstractArray:
    """Ghost witness for solve_nnls."""
    _ = (A, b)
    return AbstractArray(shape=A.shape, dtype=A.dtype)

