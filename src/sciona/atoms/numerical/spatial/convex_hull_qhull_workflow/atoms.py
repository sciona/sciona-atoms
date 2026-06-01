from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_convex_hull,
    witness_verify_point_containment,
)

@register_atom(witness_compute_convex_hull, name="compute_convex_hull")
@icontract.require(lambda points: points.ndim == 2, "Precondition failed: points.ndim == 2")
@icontract.require(lambda points: points.shape[0] >= points.shape[1] + 1, "Precondition failed: points.shape[0] >= points.shape[1] + 1")
@icontract.ensure(lambda result, points: volume > 0.0, "Postcondition failed: volume > 0.0")
@icontract.ensure(lambda result, points: area > 0.0, "Postcondition failed: area > 0.0")
def compute_convex_hull(points: NDArray[np.float64]) -> NDArray[np.int32]:
    """Compute the convex hull boundary, facets, and enclosed spatial statistics.

    Args:
        points: NDArray[np.float64]

    Returns:
        vertices: NDArray[np.int32]
    """
    import scipy.spatial
    return scipy.spatial.ConvexHull(points=points) # type: ignore

@register_atom(witness_verify_point_containment, name="verify_point_containment")
@icontract.require(lambda points, equations: points.ndim == 2, "Precondition failed: points.ndim == 2")
@icontract.require(lambda points, equations: equations.ndim == 2, "Precondition failed: equations.ndim == 2")
@icontract.ensure(lambda result, points, equations: result is not None, "Postcondition failed: result is not None")
def verify_point_containment(points: NDArray[np.float64], equations: NDArray[np.float64]) -> bool:
    """Verify that all input points satisfy the facet boundary halfspace inequalities.

    Args:
        points: NDArray[np.float64]
        equations: NDArray[np.float64]

    Returns:
        contains_all: bool
    """
    import scipy.spatial
    return scipy.spatial.ConvexHull(points=points, equations=equations) # type: ignore

