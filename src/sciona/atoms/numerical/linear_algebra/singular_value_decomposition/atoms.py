from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_dense_svd,
)

@register_atom(witness_dense_svd, name="dense_svd")
@icontract.require(lambda A, full_matrices: A.ndim == 2, "Precondition failed: A.ndim == 2")
@icontract.ensure(lambda result, A, full_matrices: s.ndim == 1, "Postcondition failed: s.ndim == 1")
def dense_svd(A: NDArray[np.float64 | np.complex128], full_matrices: bool = None) -> NDArray[np.float64 | np.complex128]:
    """Compute Singular Value Decomposition of A.

    Args:
        A: NDArray[np.float64 | np.complex128]
        full_matrices: Default False

    Returns:
        U: NDArray[np.float64 | np.complex128]
    """
    import scipy.linalg
    return scipy.linalg.svd(A=A, full_matrices=full_matrices) # type: ignore

