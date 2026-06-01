from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_pairwise_cdist,
)

@register_atom(witness_compute_pairwise_cdist, name="compute_pairwise_cdist")
@icontract.require(lambda XA, XB, metric: XA.ndim == 2, "Precondition failed: XA.ndim == 2")
@icontract.require(lambda XA, XB, metric: XB.ndim == 2, "Precondition failed: XB.ndim == 2")
@icontract.require(lambda XA, XB, metric: XA.shape[1] == XB.shape[1], "Precondition failed: XA.shape[1] == XB.shape[1]")
@icontract.ensure(lambda result, XA, XB, metric: dm.shape == (XA.shape[0], XB.shape[0]), "Postcondition failed: dm.shape == (XA.shape[0], XB.shape[0])")
@icontract.ensure(lambda result, XA, XB, metric: np.all(dm >= 0.0), "Postcondition failed: np.all(dm >= 0.0)")
def compute_pairwise_cdist(XA: NDArray[np.float64], XB: NDArray[np.float64], metric: str = None) -> NDArray[np.float64]:
    """Compute pairwise distance between two sets of vectors in a vectorized manner.

    Args:
        XA: NDArray[np.float64]
        XB: NDArray[np.float64]
        metric: str

    Returns:
        dm: NDArray[np.float64]
    """
    import scipy.spatial.distance
    return scipy.spatial.distance.cdist(XA=XA, XB=XB, metric=metric) # type: ignore

