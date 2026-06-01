from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_schur_decompose_matrix,
)

@register_atom(witness_schur_decompose_matrix, name="schur_decompose_matrix")
@icontract.require(lambda A, output_type: A.ndim == 2, "Precondition failed: A.ndim == 2")
@icontract.require(lambda A, output_type: A.shape[0] == A.shape[1], "Precondition failed: A.shape[0] == A.shape[1]")
@icontract.ensure(lambda result, A, output_type: T.shape == A.shape, "Postcondition failed: T.shape == A.shape")
@icontract.ensure(lambda result, A, output_type: Q.shape == A.shape, "Postcondition failed: Q.shape == A.shape")
def schur_decompose_matrix(A: NDArray[np.float64 | np.complex128], output_type: str = None) -> NDArray[np.float64 | np.complex128]:
    """Compute Schur form of square matrix A.

    Args:
        A: NDArray[np.float64 | np.complex128]
        output_type: real or complex

    Returns:
        T: NDArray[np.float64 | np.complex128]
    """
    import scipy.linalg
    return scipy.linalg.schur(A=A, output_type=output_type) # type: ignore

