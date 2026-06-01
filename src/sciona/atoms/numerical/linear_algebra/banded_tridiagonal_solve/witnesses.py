from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_solve_banded_system(l_and_u: AbstractScalar | int, ab: AbstractArray, b: AbstractArray) -> AbstractArray:
    """Ghost witness for solve_banded_system."""
    _ = (l_and_u, ab, b)
    return AbstractArray(shape=ab.shape, dtype=ab.dtype)

