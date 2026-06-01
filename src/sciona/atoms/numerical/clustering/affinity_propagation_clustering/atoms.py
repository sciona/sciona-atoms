from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_AP_message_update,
    witness_extract_exemplars,
)

@register_atom(witness_AP_message_update, name="AP_message_update")
@icontract.require(lambda S, R, A, damping: damping >= 0.5, "Precondition failed: damping >= 0.5")
@icontract.require(lambda S, R, A, damping: damping < 1.0, "Precondition failed: damping < 1.0")
@icontract.ensure(lambda result, S, R, A, damping: R_new.shape == S.shape, "Postcondition failed: R_new.shape == S.shape")
def AP_message_update(S: NDArray[np.float64], R: NDArray[np.float64], A: NDArray[np.float64], damping: float) -> NDArray[np.float64]:
    """Run message-passing iterations to update responsibility and availability matrices with damping.

    Args:
        S: NDArray[np.float64]
        R: NDArray[np.float64]
        A: NDArray[np.float64]
        damping: float

    Returns:
        R_new: NDArray[np.float64]
    """
    import sklearn.cluster
    return sklearn.cluster.AffinityPropagation(S=S, R=R, A=A, damping=damping) # type: ignore

@register_atom(witness_extract_exemplars, name="extract_exemplars")
@icontract.require(lambda R, A: R.shape == A.shape, "Precondition failed: R.shape == A.shape")
@icontract.ensure(lambda result, R, A: labels.shape[0] == R.shape[0], "Postcondition failed: labels.shape[0] == R.shape[0]")
def extract_exemplars(R: NDArray[np.float64], A: NDArray[np.float64]) -> NDArray[np.int64]:
    """Identify exemplars from converged message matrices and assign labels.

    Args:
        R: NDArray[np.float64]
        A: NDArray[np.float64]

    Returns:
        exemplars: NDArray[np.int64]
    """
    import sklearn.cluster
    return sklearn.cluster.AffinityPropagation(R=R, A=A) # type: ignore

