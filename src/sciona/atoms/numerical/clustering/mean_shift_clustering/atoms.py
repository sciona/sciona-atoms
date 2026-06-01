from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_shift_step,
    witness_extract_modes,
)

@register_atom(witness_shift_step, name="shift_step")
@icontract.require(lambda X, current_positions, bandwidth: bandwidth > 0.0, "Precondition failed: bandwidth > 0.0")
@icontract.ensure(lambda result, X, current_positions, bandwidth: new_positions.shape == current_positions.shape, "Postcondition failed: new_positions.shape == current_positions.shape")
def shift_step(X: NDArray[np.float64], current_positions: NDArray[np.float64], bandwidth: float) -> NDArray[np.float64]:
    """Perform a single vectorized shift of coordinates towards local density peaks.

    Args:
        X: NDArray[np.float64]
        current_positions: NDArray[np.float64]
        bandwidth: float

    Returns:
        new_positions: NDArray[np.float64]
    """
    import sklearn.cluster
    return sklearn.cluster.MeanShift(X=X, current_positions=current_positions, bandwidth=bandwidth) # type: ignore

@register_atom(witness_extract_modes, name="extract_modes")
@icontract.require(lambda converged_positions, bandwidth: bandwidth > 0.0, "Precondition failed: bandwidth > 0.0")
@icontract.ensure(lambda result, converged_positions, bandwidth: labels.shape[0] == converged_positions.shape[0], "Postcondition failed: labels.shape[0] == converged_positions.shape[0]")
def extract_modes(converged_positions: NDArray[np.float64], bandwidth: float) -> NDArray[np.float64]:
    """Consolidate shifted points into discrete mode centroids and labels.

    Args:
        converged_positions: NDArray[np.float64]
        bandwidth: float

    Returns:
        centroids: NDArray[np.float64]
    """
    import sklearn.cluster
    return sklearn.cluster.MeanShift(converged_positions=converged_positions, bandwidth=bandwidth) # type: ignore

