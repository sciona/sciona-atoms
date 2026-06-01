from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_validate_and_sort_descriptive_inputs,
    witness_compute_robust_location,
    witness_compute_robust_scale,
)

@register_atom(witness_validate_and_sort_descriptive_inputs, name="validate_and_sort_descriptive_inputs")
@icontract.require(lambda data: data.ndim == 1, "Precondition failed: data.ndim == 1")
@icontract.require(lambda data: len(data) >= 3, "Precondition failed: len(data) >= 3")
@icontract.require(lambda data: np.all(np.isfinite(data)), "Precondition failed: np.all(np.isfinite(data))")
@icontract.ensure(lambda result, data: len(result) == len(data), "Postcondition failed: len(result) == len(data)")
@icontract.ensure(lambda result, data: np.all(np.diff(result) >= 0), "Postcondition failed: np.all(np.diff(result) >= 0)")
def validate_and_sort_descriptive_inputs(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Verify array dimensions, finiteness, and sort data in ascending order.

    Args:
        data: 1D finite array

    Returns:
        sorted_data: NDArray[np.float64]
    """
    import numpy
    return numpy.sort(data=data) # type: ignore

@register_atom(witness_compute_robust_location, name="compute_robust_location")
@icontract.require(lambda sorted_data, trim_ratio: trim_ratio >= 0.0, "Precondition failed: trim_ratio >= 0.0")
@icontract.require(lambda sorted_data, trim_ratio: trim_ratio < 0.5, "Precondition failed: trim_ratio < 0.5")
@icontract.ensure(lambda result, sorted_data, trim_ratio: 'median' in result, "Postcondition failed: 'median' in result")
@icontract.ensure(lambda result, sorted_data, trim_ratio: 'trimmed_mean' in result, "Postcondition failed: 'trimmed_mean' in result")
def compute_robust_location(sorted_data: NDArray[np.float64], trim_ratio: float = None) -> float:
    """Calculate median, trimmed mean, and Winsorized mean.

    Args:
        sorted_data: NDArray[np.float64]
        trim_ratio: 0.0 <= trim_ratio < 0.5

    Returns:
        location_metrics: dict[str, float]
    """
    import scipy.stats
    return scipy.stats.tmean(sorted_data=sorted_data, trim_ratio=trim_ratio) # type: ignore

@register_atom(witness_compute_robust_scale, name="compute_robust_scale")
@icontract.require(lambda sorted_data: sorted_data is not None, "Precondition failed: sorted_data is not None")
@icontract.ensure(lambda result, sorted_data: result['iqr'] >= 0.0, "Postcondition failed: result['iqr'] >= 0.0")
@icontract.ensure(lambda result, sorted_data: result['mad'] >= 0.0, "Postcondition failed: result['mad'] >= 0.0")
def compute_robust_scale(sorted_data: NDArray[np.float64]) -> float:
    """Calculate Interquartile Range (IQR) and Median Absolute Deviation (MAD).

    Args:
        sorted_data: NDArray[np.float64]

    Returns:
        scale_metrics: dict[str, float]
    """
    import scipy.stats
    return scipy.stats.iqr(sorted_data=sorted_data) # type: ignore

