from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_local_reachability_density,
    witness_score_outlier_factors,
)

@register_atom(witness_compute_local_reachability_density, name="compute_local_reachability_density")
@icontract.require(lambda distances, indices: distances.shape == indices.shape, "Precondition failed: distances.shape == indices.shape")
@icontract.ensure(lambda result, distances, indices: lrd.shape[0] == distances.shape[0], "Postcondition failed: lrd.shape[0] == distances.shape[0]")
def compute_local_reachability_density(distances: NDArray[np.float64], indices: NDArray[np.int64]) -> NDArray[np.float64]:
    """Calculate the reachability distances and local densities of points relative to neighbors.

    Args:
        distances: NDArray[np.float64]
        indices: NDArray[np.int64]

    Returns:
        lrd: NDArray[np.float64]
    """
    import sklearn.neighbors
    return sklearn.neighbors.LocalOutlierFactor(distances=distances, indices=indices) # type: ignore

@register_atom(witness_score_outlier_factors, name="score_outlier_factors")
@icontract.require(lambda indices, lrd: indices.shape[0] == lrd.shape[0], "Precondition failed: indices.shape[0] == lrd.shape[0]")
@icontract.ensure(lambda result, indices, lrd: lof_scores.shape[0] == lrd.shape[0], "Postcondition failed: lof_scores.shape[0] == lrd.shape[0]")
def score_outlier_factors(indices: NDArray[np.int64], lrd: NDArray[np.float64]) -> NDArray[np.float64]:
    """Assemble LOF scores by computing the average local density ratio of neighbors.

    Args:
        indices: NDArray[np.int64]
        lrd: NDArray[np.float64]

    Returns:
        lof_scores: NDArray[np.float64]
    """
    import sklearn.neighbors
    return sklearn.neighbors.LocalOutlierFactor(indices=indices, lrd=lrd) # type: ignore

