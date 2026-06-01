from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_find_correspondences,
    witness_estimate_rigid_transform,
)

@register_atom(witness_find_correspondences, name="find_correspondences")
@icontract.require(lambda source, target: source.ndim == 2, "Precondition failed: source.ndim == 2")
@icontract.require(lambda source, target: target.ndim == 2, "Precondition failed: target.ndim == 2")
@icontract.require(lambda source, target: source.shape[1] == target.shape[1], "Precondition failed: source.shape[1] == target.shape[1]")
@icontract.ensure(lambda result, source, target: matched_target.shape == source.shape, "Postcondition failed: matched_target.shape == source.shape")
def find_correspondences(source: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    """Use spatial indices to associate each point in the source cloud to its nearest neighbor in the target cloud.

    Args:
        source: NDArray[np.float64]
        target: NDArray[np.float64]

    Returns:
        matched_target: NDArray[np.float64]
    """
    import scipy.spatial
    return scipy.spatial.cKDTree(source=source, target=target) # type: ignore

@register_atom(witness_estimate_rigid_transform, name="estimate_rigid_transform")
@icontract.require(lambda source, matched_target: source.shape == matched_target.shape, "Precondition failed: source.shape == matched_target.shape")
@icontract.ensure(lambda result, source, matched_target: R.shape == (source.shape[1], source.shape[1]), "Postcondition failed: R.shape == (source.shape[1], source.shape[1])")
@icontract.ensure(lambda result, source, matched_target: t.shape[0] == source.shape[1], "Postcondition failed: t.shape[0] == source.shape[1]")
def estimate_rigid_transform(source: NDArray[np.float64], matched_target: NDArray[np.float64]) -> NDArray[np.float64]:
    """Use centered SVD solvers to compute the optimal rotation and translation aligning matched coordinates.

    Args:
        source: NDArray[np.float64]
        matched_target: NDArray[np.float64]

    Returns:
        R: NDArray[np.float64]
    """
    import scipy.spatial
    return scipy.spatial.procrustes(source=source, matched_target=matched_target) # type: ignore

