from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_check_and_factorize_metric,
    witness_solve_generalized_symmetric,
)

@register_atom(witness_check_and_factorize_metric, name="check_and_factorize_metric")
@icontract.require(lambda B: B.ndim == 2, "Precondition failed: B.ndim == 2")
@icontract.require(lambda B: B.shape[0] == B.shape[1], "Precondition failed: B.shape[0] == B.shape[1]")
@icontract.ensure(lambda result, B: result is not None, "Postcondition failed: result is not None")
def check_and_factorize_metric(B: NDArray[np.float64]) -> tuple[NDArray[np.float64], bool]:
    """Verify and compute Cholesky factorization of positive definite matrix B.

    Args:
        B: NDArray[np.float64]

    Returns:
        c_factor: tuple[NDArray[np.float64], bool]
    """
    import scipy.linalg
    return scipy.linalg.cho_factor(B=B) # type: ignore

@register_atom(witness_solve_generalized_symmetric, name="solve_generalized_symmetric")
@icontract.require(lambda A, c_factor: A.shape == c_factor[0].shape, "Precondition failed: A.shape == c_factor[0].shape")
@icontract.ensure(lambda result, A, c_factor: eigenvalues.ndim == 1, "Postcondition failed: eigenvalues.ndim == 1")
def solve_generalized_symmetric(A: NDArray[np.float64], c_factor: tuple[NDArray[np.float64], bool]) -> NDArray[np.float64]:
    """Compute generalized symmetric-definite eigenvalues using Cholesky factor.

    Args:
        A: NDArray[np.float64]
        c_factor: tuple[NDArray[np.float64], bool]

    Returns:
        eigenvalues: NDArray[np.float64]
    """
    import scipy.linalg
    return scipy.linalg.eigh(A=A, c_factor=c_factor) # type: ignore

