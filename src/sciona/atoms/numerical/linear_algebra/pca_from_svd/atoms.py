from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_center_data,
    witness_pca_svd_decompose,
    witness_calculate_pca_variance,
)

@register_atom(witness_center_data, name="center_data")
@icontract.require(lambda X: X.ndim == 2)
@icontract.ensure(lambda result, X: result[0].shape == X.shape)
@icontract.ensure(lambda result, X: result[1].shape == (X.shape[1],))
@icontract.ensure(lambda result: np.all(np.isfinite(result[0])))
@icontract.ensure(lambda result: np.all(np.isfinite(result[1])))
def center_data(X: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute mean and center the input matrix.

    Parameters
    ----------
    X : NDArray[np.float64]
        Matrix of shape (m, n).

    Returns
    -------
    X_centered : NDArray[np.float64]
        Centered matrix of shape (m, n).
    mean : NDArray[np.float64]
        Row mean of shape (n,).
    """
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    return X_centered, mean

@register_atom(witness_pca_svd_decompose, name="pca_svd_decompose")
@icontract.require(lambda X_centered, n_components: X_centered.ndim == 2)
@icontract.require(lambda X_centered, n_components: n_components > 0)
@icontract.require(lambda X_centered, n_components: n_components <= min(X_centered.shape))
@icontract.ensure(lambda result, n_components, X_centered: result[0].shape == (n_components, X_centered.shape[1]))
@icontract.ensure(lambda result, n_components: result[1].shape == (n_components,))
@icontract.ensure(lambda result: np.all(np.isfinite(result[0])))
@icontract.ensure(lambda result: np.all(np.isfinite(result[1])))
def pca_svd_decompose(
    X_centered: NDArray[np.float64],
    n_components: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Perform SVD decomposition on centered data and compute loadings.

    Parameters
    ----------
    X_centered : NDArray[np.float64]
        Centered matrix of shape (m, n).
    n_components : int
        Number of components to keep.

    Returns
    -------
    components : NDArray[np.float64]
        Principal components (loadings) of shape (n_components, n).
    singular_values : NDArray[np.float64]
        Singular values of shape (n_components,).
    """
    import scipy.linalg
    U, s, Vt = scipy.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components]
    singular_values = s[:n_components]
    return components, singular_values

@register_atom(witness_calculate_pca_variance, name="calculate_pca_variance")
@icontract.require(lambda singular_values, n_samples: singular_values.ndim == 1)
@icontract.require(lambda singular_values, n_samples: n_samples > 1)
@icontract.ensure(lambda result, singular_values: result.shape == singular_values.shape)
@icontract.ensure(lambda result: np.all(np.isfinite(result)))
def calculate_pca_variance(
    singular_values: NDArray[np.float64],
    n_samples: int,
) -> NDArray[np.float64]:
    """Calculate explained variance metrics.

    Parameters
    ----------
    singular_values : NDArray[np.float64]
        Singular values of shape (n_components,).
    n_samples : int
        Number of samples used in PCA.

    Returns
    -------
    explained_variance : NDArray[np.float64]
        Explained variance of shape (n_components,).
    """
    explained_variance = (singular_values ** 2) / (n_samples - 1)
    return explained_variance


