from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_iterate_lsmr,
)

@register_atom(witness_iterate_lsmr, name="iterate_lsmr")
@icontract.require(lambda A, b, damp, atol, btol: b.ndim == 1, "Precondition failed: b.ndim == 1")
@icontract.ensure(lambda result, A, b, damp, atol, btol: result is not None, "Postcondition failed: result is not None")
def iterate_lsmr(A: Any, b: NDArray[np.float64], damp: float = None, atol: float = None, btol: float = None) -> NDArray[np.float64]:
    """Run LSMR bidiagonalization iterations.

    Args:
        A: scipy.sparse.linalg.LinearOperator
        b: NDArray[np.float64]
        damp: float
        atol: float
        btol: float

    Returns:
        x: NDArray[np.float64]
    """
    import scipy.sparse.linalg
    return scipy.sparse.linalg.lsmr(A=A, b=b, damp=damp, atol=atol, btol=btol) # type: ignore

