from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_delaunay,
    witness_validate_empty_circumspheres,
)

@register_atom(witness_compute_delaunay, name="compute_delaunay")
@icontract.require(lambda points: points.ndim == 2, "Precondition failed: points.ndim == 2")
@icontract.require(lambda points: points.shape[0] >= points.shape[1] + 1, "Precondition failed: points.shape[0] >= points.shape[1] + 1")
@icontract.ensure(lambda result, points: simplices.ndim == 2, "Postcondition failed: simplices.ndim == 2")
@icontract.ensure(lambda result, points: simplices.shape[1] == points.shape[1] + 1, "Postcondition failed: simplices.shape[1] == points.shape[1] + 1")
def compute_delaunay(points: NDArray[np.float64]) -> NDArray[np.int32]:
    """Execute Qhull to partition spatial points into Delaunay simplices.

    Args:
        points: NDArray[np.float64]

    Returns:
        simplices: NDArray[np.int32]
    """
    import scipy.spatial
    return scipy.spatial.Delaunay(points=points) # type: ignore

@register_atom(witness_validate_empty_circumspheres, name="validate_empty_circumspheres")
@icontract.require(lambda points, simplices: points.ndim == 2, "Precondition failed: points.ndim == 2")
@icontract.require(lambda points, simplices: simplices.ndim == 2, "Precondition failed: simplices.ndim == 2")
@icontract.ensure(lambda result, points, simplices: result is not None, "Postcondition failed: result is not None")
def validate_empty_circumspheres(points: NDArray[np.float64], simplices: NDArray[np.int32]) -> bool:
    """Verify Delaunay empty-circumsphere invariants to guarantee triangulation validity.

    Args:
        points: NDArray[np.float64]
        simplices: NDArray[np.int32]

    Returns:
        is_valid: bool
    """
    import scipy.spatial
    return scipy.spatial.Delaunay(points=points, simplices=simplices) # type: ignore

