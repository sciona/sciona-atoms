from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_ilut_factors(A: AbstractScalar | Any, drop_tol: AbstractScalar | float) -> AbstractScalar:
    """Ghost witness for compute_ilut_factors."""
    _ = (A, drop_tol)
    return AbstractScalar(dtype="float64")

def witness_create_preconditioner_operator(ilu_obj: AbstractScalar | Any) -> AbstractScalar:
    """Ghost witness for create_preconditioner_operator."""
    _ = (ilu_obj)
    return AbstractScalar(dtype="float64")

