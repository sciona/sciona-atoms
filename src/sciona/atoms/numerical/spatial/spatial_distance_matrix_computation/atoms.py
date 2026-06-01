from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_invert_covariance,
    witness_compute_mahalanobis_cdist,
)

@register_atom(witness_invert_covariance, name="invert_covariance")
@icontract.require(lambda cov, ridge: cov.ndim == 2, "Precondition failed: cov.ndim == 2")
@icontract.require(lambda cov, ridge: cov.shape[0] == cov.shape[1], "Precondition failed: cov.shape[0] == cov.shape[1]")
@icontract.ensure(lambda result, cov, ridge: inv_cov.shape == cov.shape, "Postcondition failed: inv_cov.shape == cov.shape")
def invert_covariance(cov: NDArray[np.float64], ridge: float = None) -> NDArray[np.float64]:
    """Verify and invert a covariance matrix with ridge regularization.

    Args:
        cov: NDArray[np.float64]
        ridge: float

    Returns:
        inv_cov: NDArray[np.float64]
    """
    import scipy.linalg
    return scipy.linalg.inv(cov=cov, ridge=ridge) # type: ignore

@register_atom(witness_compute_mahalanobis_cdist, name="compute_mahalanobis_cdist")
@icontract.require(lambda XA, XB, inv_cov: XA.shape[1] == inv_cov.shape[0], "Precondition failed: XA.shape[1] == inv_cov.shape[0]")
@icontract.ensure(lambda result, XA, XB, inv_cov: dm.shape == (XA.shape[0], XB.shape[0]), "Postcondition failed: dm.shape == (XA.shape[0], XB.shape[0])")
def compute_mahalanobis_cdist(XA: NDArray[np.float64], XB: NDArray[np.float64], inv_cov: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute pairwise Mahalanobis distances using the inverse covariance matrix.

    Args:
        XA: NDArray[np.float64]
        XB: NDArray[np.float64]
        inv_cov: NDArray[np.float64]

    Returns:
        dm: NDArray[np.float64]
    """
    import scipy.spatial.distance
    return scipy.spatial.distance.cdist(XA=XA, XB=XB, inv_cov=inv_cov) # type: ignore

