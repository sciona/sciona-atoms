from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_evaluate_gauss_kronrod_15(func: AbstractScalar | float, a: AbstractScalar | float, b: AbstractScalar | float) -> AbstractScalar:
    """Ghost witness for evaluate_gauss_kronrod_15."""
    _ = (func, a, b)
    return AbstractScalar(dtype="float64")

def witness_adaptive_subdivision_loop(func: AbstractScalar | float, a: AbstractScalar | float, b: AbstractScalar | float, epsabs: AbstractScalar | float, epsrel: AbstractScalar | float, limit: AbstractScalar | int) -> AbstractScalar:
    """Ghost witness for adaptive_subdivision_loop."""
    _ = (func, a, b, epsabs, epsrel, limit)
    return AbstractScalar(dtype="float64")

