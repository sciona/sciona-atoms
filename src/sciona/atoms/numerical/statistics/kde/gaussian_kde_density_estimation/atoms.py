from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_select_kde_bandwidth,
    witness_evaluate_kde_density,
)

@register_atom(witness_select_kde_bandwidth, name="select_kde_bandwidth")
@icontract.require(lambda data, bw_method: data.ndim == 2, "Precondition failed: data.ndim == 2")
@icontract.require(lambda data, bw_method: data.shape[1] > data.shape[0], "Precondition failed: data.shape[1] > data.shape[0]")
@icontract.ensure(lambda result, data, bw_method: result > 0.0, "Postcondition failed: result > 0.0")
def select_kde_bandwidth(data: NDArray[np.float64], bw_method: str = None) -> float:
    """Determine bandwidth factor according to Scott or Silverman rules, verifying dataset dimensions.

    Args:
        data: 2D array (dims, samples)
        bw_method: str

    Returns:
        factor: float
    """
    import scipy.stats.gaussian_kde
    return scipy.stats.gaussian_kde.scotts_factor(data=data, bw_method=bw_method) # type: ignore

@register_atom(witness_evaluate_kde_density, name="evaluate_kde_density")
@icontract.require(lambda data, eval_points, factor: data.shape[0] == eval_points.shape[0], "Precondition failed: data.shape[0] == eval_points.shape[0]")
@icontract.require(lambda data, eval_points, factor: factor > 0.0, "Precondition failed: factor > 0.0")
@icontract.ensure(lambda result, data, eval_points, factor: density.ndim == 1, "Postcondition failed: density.ndim == 1")
@icontract.ensure(lambda result, data, eval_points, factor: density.shape[0] == eval_points.shape[1], "Postcondition failed: density.shape[0] == eval_points.shape[1]")
@icontract.ensure(lambda result, data, eval_points, factor: np.all(density >= 0.0), "Postcondition failed: np.all(density >= 0.0)")
def evaluate_kde_density(data: NDArray[np.float64], eval_points: NDArray[np.float64], factor: float) -> NDArray[np.float64]:
    """Perform multi-dimensional kernel evaluation using Cholesky factorized covariance and the bandwidth.

    Args:
        data: NDArray[np.float64]
        eval_points: NDArray[np.float64]
        factor: float

    Returns:
        density: NDArray[np.float64]
    """
    import scipy.stats.gaussian_kde
    return scipy.stats.gaussian_kde._evaluate(data=data, eval_points=eval_points, factor=factor) # type: ignore

