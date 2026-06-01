from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_polar_decomposition,
)

@register_atom(witness_compute_polar_decomposition, name="compute_polar_decomposition")
@icontract.require(lambda A: A.ndim == 2, "Precondition failed: A.ndim == 2")
@icontract.require(lambda A: A.shape[0] == A.shape[1], "Precondition failed: A.shape[0] == A.shape[1]")
@icontract.ensure(lambda result, A: U.shape == A.shape, "Postcondition failed: U.shape == A.shape")
@icontract.ensure(lambda result, A: P.shape == A.shape, "Postcondition failed: P.shape == A.shape")
def compute_polar_decomposition(A: NDArray[np.float64 | np.complex128]) -> NDArray[np.float64 | np.complex128]:
    """Decompose matrix into unitary rotation and symmetric stretch factors.

    Args:
        A: NDArray[np.float64 | np.complex128]

    Returns:
        U: NDArray[np.float64 | np.complex128]
    """
    import scipy.linalg
    return scipy.linalg.polar(A=A) # type: ignore

