from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_iterate_lsmr(A: AbstractScalar | Any, b: AbstractArray, damp: AbstractScalar | float, atol: AbstractScalar | float, btol: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for iterate_lsmr."""
    _ = (A, b, damp, atol, btol)
    return AbstractArray(shape=b.shape, dtype=b.dtype)

