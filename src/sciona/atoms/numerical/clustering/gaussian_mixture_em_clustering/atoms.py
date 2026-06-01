from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gmm_e_step,
    witness_gmm_m_step,
)

@register_atom(witness_gmm_e_step, name="gmm_e_step")
@icontract.require(lambda X, means, covariances, weights: X.shape[1] == means.shape[1], "Precondition failed: X.shape[1] == means.shape[1]")
@icontract.ensure(lambda result, X, means, covariances, weights: responsibilities.shape == (X.shape[0], means.shape[0]), "Postcondition failed: responsibilities.shape == (X.shape[0], means.shape[0])")
def gmm_e_step(X: NDArray[np.float64], means: NDArray[np.float64], covariances: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute component responsibilities (posterior probabilities) for all data records.

    Args:
        X: NDArray[np.float64]
        means: NDArray[np.float64]
        covariances: NDArray[np.float64]
        weights: NDArray[np.float64]

    Returns:
        responsibilities: NDArray[np.float64]
    """
    import sklearn.mixture
    return sklearn.mixture.GaussianMixture(X=X, means=means, covariances=covariances, weights=weights) # type: ignore

@register_atom(witness_gmm_m_step, name="gmm_m_step")
@icontract.require(lambda X, responsibilities, covariance_type: responsibilities.ndim == 2, "Precondition failed: responsibilities.ndim == 2")
@icontract.ensure(lambda result, X, responsibilities, covariance_type: means.shape[0] == responsibilities.shape[1], "Postcondition failed: means.shape[0] == responsibilities.shape[1]")
def gmm_m_step(X: NDArray[np.float64], responsibilities: NDArray[np.float64], covariance_type: str) -> NDArray[np.float64]:
    """Recalculate Gaussian mixture parameters (means, covariances, mixing weights) using responsibilities.

    Args:
        X: NDArray[np.float64]
        responsibilities: NDArray[np.float64]
        covariance_type: str

    Returns:
        means: NDArray[np.float64]
    """
    import sklearn.mixture
    return sklearn.mixture.GaussianMixture(X=X, responsibilities=responsibilities, covariance_type=covariance_type) # type: ignore

