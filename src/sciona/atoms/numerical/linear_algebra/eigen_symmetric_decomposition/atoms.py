from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_symmetric_eigenvalues,
)

@register_atom(witness_compute_symmetric_eigenvalues, name="compute_symmetric_eigenvalues")
@icontract.require(lambda A, eigvals_only: A.ndim == 2, "Precondition failed: A.ndim == 2")
@icontract.require(lambda A, eigvals_only: A.shape[0] == A.shape[1], "Precondition failed: A.shape[0] == A.shape[1]")
@icontract.ensure(lambda result, A, eigvals_only: eigenvalues.ndim == 1, "Postcondition failed: eigenvalues.ndim == 1")
def compute_symmetric_eigenvalues(A: NDArray[np.float64 | np.complex128], eigvals_only: bool = None) -> NDArray[np.float64]:
    """Compute real eigenvalues and optional eigenvectors of a symmetric matrix.

    Args:
        A: NDArray[np.float64 | np.complex128]
        eigvals_only: Default False

    Returns:
        eigenvalues: NDArray[np.float64]
    """
    import scipy.linalg
    return scipy.linalg.eigh(A=A, eigvals_only=eigvals_only) # type: ignore

