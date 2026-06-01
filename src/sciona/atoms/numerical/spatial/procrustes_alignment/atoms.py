from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_center_and_normalize,
    witness_compute_optimal_rotation,
    witness_apply_transform_and_measure,
)

@register_atom(witness_center_and_normalize, name="center_and_normalize")
@icontract.require(lambda data: data.ndim == 2, "Precondition failed: data.ndim == 2")
@icontract.ensure(lambda result, data: np.allclose(centered.mean(axis=0), 0.0), "Postcondition failed: np.allclose(centered.mean(axis=0), 0.0)")
def center_and_normalize(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Center a point cloud to its mean and scale its root-mean-square distance to 1.0.

    Args:
        data: NDArray[np.float64]

    Returns:
        centered: NDArray[np.float64]
    """
    import scipy.spatial
    return scipy.spatial.procrustes(data=data) # type: ignore

@register_atom(witness_compute_optimal_rotation, name="compute_optimal_rotation")
@icontract.require(lambda standardized_ref, standardized_src: standardized_ref.shape == standardized_src.shape, "Precondition failed: standardized_ref.shape == standardized_src.shape")
@icontract.ensure(lambda result, standardized_ref, standardized_src: rotation.shape == (standardized_ref.shape[1], standardized_ref.shape[1]), "Postcondition failed: rotation.shape == (standardized_ref.shape[1], standardized_ref.shape[1])")
def compute_optimal_rotation(standardized_ref: NDArray[np.float64], standardized_src: NDArray[np.float64]) -> NDArray[np.float64]:
    """Use SVD to compute the optimal orthogonal rotation aligning two standardized point sets.

    Args:
        standardized_ref: NDArray[np.float64]
        standardized_src: NDArray[np.float64]

    Returns:
        rotation: NDArray[np.float64]
    """
    import scipy.spatial
    return scipy.spatial.procrustes(standardized_ref=standardized_ref, standardized_src=standardized_src) # type: ignore

@register_atom(witness_apply_transform_and_measure, name="apply_transform_and_measure")
@icontract.require(lambda standardized_ref, standardized_src, rotation: rotation.ndim == 2, "Precondition failed: rotation.ndim == 2")
@icontract.ensure(lambda result, standardized_ref, standardized_src, rotation: disparity >= 0.0, "Postcondition failed: disparity >= 0.0")
def apply_transform_and_measure(standardized_ref: NDArray[np.float64], standardized_src: NDArray[np.float64], rotation: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotate the source dataset, scale it, and compute the residual squared error disparity.

    Args:
        standardized_ref: NDArray[np.float64]
        standardized_src: NDArray[np.float64]
        rotation: NDArray[np.float64]

    Returns:
        aligned: NDArray[np.float64]
    """
    import scipy.spatial
    return scipy.spatial.procrustes(standardized_ref=standardized_ref, standardized_src=standardized_src, rotation=rotation) # type: ignore

