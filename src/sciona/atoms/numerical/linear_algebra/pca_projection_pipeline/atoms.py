from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_center_features,
    witness_factorize_svd,
    witness_project_coordinates,
)

@register_atom(witness_center_features, name="center_features")
@icontract.require(lambda X: X.ndim == 2, "Precondition failed: X.ndim == 2")
@icontract.ensure(lambda result, X: np.allclose(centered.mean(axis=0), 0.0), "Postcondition failed: np.allclose(centered.mean(axis=0), 0.0)")
def center_features(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Center columns of a matrix to have zero mean.

    Args:
        X: NDArray[np.float64]

    Returns:
        centered: NDArray[np.float64]
    """
    import sklearn.decomposition
    return sklearn.decomposition.PCA(X=X) # type: ignore

@register_atom(witness_factorize_svd, name="factorize_svd")
@icontract.require(lambda centered, n_components: n_components <= min(centered.shape), "Precondition failed: n_components <= min(centered.shape)")
@icontract.ensure(lambda result, centered, n_components: components.shape == (n_components, centered.shape[1]), "Postcondition failed: components.shape == (n_components, centered.shape[1])")
def factorize_svd(centered: NDArray[np.float64], n_components: int) -> NDArray[np.float64]:
    """Execute SVD on centered matrix to extract principal component axes and singular values.

    Args:
        centered: NDArray[np.float64]
        n_components: int

    Returns:
        components: NDArray[np.float64]
    """
    import scipy.linalg
    return scipy.linalg.svd(centered=centered, n_components=n_components) # type: ignore

@register_atom(witness_project_coordinates, name="project_coordinates")
@icontract.require(lambda centered, components: centered.shape[1] == components.shape[1], "Precondition failed: centered.shape[1] == components.shape[1]")
@icontract.ensure(lambda result, centered, components: projected.shape == (centered.shape[0], components.shape[0]), "Postcondition failed: projected.shape == (centered.shape[0], components.shape[0])")
def project_coordinates(centered: NDArray[np.float64], components: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project centered points onto principal axes.

    Args:
        centered: NDArray[np.float64]
        components: NDArray[np.float64]

    Returns:
        projected: NDArray[np.float64]
    """
    import sklearn.decomposition.PCA
    return sklearn.decomposition.PCA.transform(centered=centered, components=components) # type: ignore

