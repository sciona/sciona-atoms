from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_build_linear_operator(shape: AbstractScalar | int, matvec: AbstractArray, rmatvec: AbstractArray) -> AbstractScalar:
    """Ghost witness for build_linear_operator."""
    _ = (shape, matvec, rmatvec)
    return AbstractScalar(dtype="float64")

def witness_verify_adjoint_relation(operator: AbstractScalar | Any) -> AbstractScalar:
    """Ghost witness for verify_adjoint_relation."""
    _ = (operator)
    return AbstractScalar(dtype="float64")

