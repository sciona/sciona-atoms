from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_calculate_high_prec_residual(A: AbstractArray, b: AbstractArray, x: AbstractArray) -> AbstractArray:
    """Ghost witness for calculate_high_prec_residual."""
    _ = (A, b, x)
    return AbstractArray(shape=A.shape, dtype=A.dtype)

def witness_apply_refinement_step(lu_and_piv: AbstractArray, r: AbstractArray, x_old: AbstractArray) -> AbstractArray:
    """Ghost witness for apply_refinement_step."""
    _ = (lu_and_piv, r, x_old)
    return AbstractArray(shape=lu_and_piv.shape, dtype=lu_and_piv.dtype)

