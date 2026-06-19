from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_extract_random_subspace_basis,
    witness_factorize_subspace_projection,
)

@register_atom(witness_extract_random_subspace_basis, name="extract_random_subspace_basis")
@icontract.require(lambda A, k, p, n_iter: A.ndim == 2)
@icontract.require(lambda A, k, p, n_iter: k > 0)
@icontract.require(lambda A, k, p, n_iter: p >= 0)
@icontract.require(lambda A, k, p, n_iter: k + p <= min(A.shape))
@icontract.require(lambda A, k, p, n_iter: n_iter >= 0)
@icontract.ensure(lambda result, A, k, p: result.shape == (A.shape[0], k + p))
@icontract.ensure(lambda result: np.all(np.isfinite(result)))
def extract_random_subspace_basis(
    A: NDArray[np.float64],
    k: int,
    p: int,
    n_iter: int,
) -> NDArray[np.float64]:
    """Form random projection basis Q for A.

    Parameters
    ----------
    A : NDArray[np.float64]
        Matrix of shape (m, n).
    k : int
        Target rank.
    p : int
        Oversampling parameter.
    n_iter : int
        Number of power iterations.

    Returns
    -------
    Q : NDArray[np.float64]
        Orthonormal basis matrix of shape (m, k + p).
    """
    from sklearn.utils.extmath import randomized_range_finder
    # Pass a default random state to ensure deterministic behavior.
    Q = randomized_range_finder(A, size=k+p, n_iter=n_iter, random_state=42)
    return Q

@register_atom(witness_factorize_subspace_projection, name="factorize_subspace_projection")
@icontract.require(lambda A, Q, k: A.ndim == 2)
@icontract.require(lambda A, Q, k: Q.ndim == 2)
@icontract.require(lambda A, Q, k: Q.shape[0] == A.shape[0])
@icontract.require(lambda A, Q, k: k > 0)
@icontract.require(lambda A, Q, k: k <= Q.shape[1])
@icontract.ensure(lambda result, A, k: result[0].shape == (A.shape[0], k))
@icontract.ensure(lambda result, k: result[1].shape == (k,))
@icontract.ensure(lambda result, A, k: result[2].shape == (k, A.shape[1]))
@icontract.ensure(lambda result: np.all(np.isfinite(result[0])))
@icontract.ensure(lambda result: np.all(np.isfinite(result[1])))
@icontract.ensure(lambda result: np.all(np.isfinite(result[2])))
def factorize_subspace_projection(
    A: NDArray[np.float64],
    Q: NDArray[np.float64],
    k: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Project A onto Q and perform core dense SVD.

    Parameters
    ----------
    A : NDArray[np.float64]
        Matrix of shape (m, n).
    Q : NDArray[np.float64]
        Projection basis matrix of shape (m, k + p).
    k : int
        Target rank (k <= Q.shape[1]).

    Returns
    -------
    U_k : NDArray[np.float64]
        Approximated left singular vectors of shape (m, k).
    s_k : NDArray[np.float64]
        Approximated singular values of shape (k,).
    Vh_k : NDArray[np.float64]
        Approximated right singular vectors of shape (k, n).
    """
    import scipy.linalg
    B = Q.T @ A
    U_B, s, Vt = scipy.linalg.svd(B, full_matrices=False)
    U = Q @ U_B
    U_k = U[:, :k]
    s_k = s[:k]
    Vh_k = Vt[:k]
    return U_k, s_k, Vh_k


