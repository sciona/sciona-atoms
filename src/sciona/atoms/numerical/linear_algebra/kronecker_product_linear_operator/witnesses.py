from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_build_kronecker_operator(A: AbstractScalar | Any, B: AbstractScalar | Any) -> AbstractScalar:
    """Ghost witness for build_kronecker_operator."""
    _ = (A, B)
    return AbstractScalar(dtype="float64")

