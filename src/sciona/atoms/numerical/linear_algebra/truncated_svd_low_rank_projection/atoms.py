from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_truncated_svds,
)

@register_atom(witness_compute_truncated_svds, name="compute_truncated_svds")
@icontract.require(lambda A, k, solver: k > 0, "Precondition failed: k > 0")
@icontract.require(lambda A, k, solver: k < min(A.shape), "Precondition failed: k < min(A.shape)")
@icontract.ensure(lambda result, A, k, solver: s_k.shape[0] == k, "Postcondition failed: s_k.shape[0] == k")
def compute_truncated_svds(A: NDArray[np.float64] | scipy.sparse.spmatrix, k: int, solver: str = None) -> NDArray[np.float64]:
    """Solve for top k singular components of a large sparse/dense matrix.

    Args:
        A: NDArray[np.float64] | scipy.sparse.spmatrix
        k: 0 < k < min(A.shape)
        solver: ARPACK or LOBPCG

    Returns:
        U_k: NDArray[np.float64]
    """
    import scipy.sparse.linalg
    return scipy.sparse.linalg.svds(A=A, k=k, solver=solver) # type: ignore

