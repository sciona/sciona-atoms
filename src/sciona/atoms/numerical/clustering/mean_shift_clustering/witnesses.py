from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_shift_step(X: AbstractArray, current_positions: AbstractArray, bandwidth: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for shift_step."""
    _ = (X, current_positions, bandwidth)
    return AbstractArray(shape=X.shape, dtype=X.dtype)

def witness_extract_modes(converged_positions: AbstractArray, bandwidth: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for extract_modes."""
    _ = (converged_positions, bandwidth)
    return AbstractArray(shape=converged_positions.shape, dtype=converged_positions.dtype)

