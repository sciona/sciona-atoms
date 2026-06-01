from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_matrix_exponential,
)

@register_atom(witness_compute_matrix_exponential, name="compute_matrix_exponential")
@icontract.require(lambda A: A.ndim == 2, "Precondition failed: A.ndim == 2")
@icontract.require(lambda A: A.shape[0] == A.shape[1], "Precondition failed: A.shape[0] == A.shape[1]")
@icontract.ensure(lambda result, A: expm_A.shape == A.shape, "Postcondition failed: expm_A.shape == A.shape")
def compute_matrix_exponential(A: NDArray[np.float64 | np.complex128]) -> NDArray[np.float64 | np.complex128]:
    """Evaluate matrix exponential using scaling and squaring with Padé approximation.

    Args:
        A: NDArray[np.float64 | np.complex128]

    Returns:
        expm_A: NDArray[np.float64 | np.complex128]
    """
    import scipy.linalg
    return scipy.linalg.expm(A=A) # type: ignore

