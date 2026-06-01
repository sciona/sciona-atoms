from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_generate_bootstrap_resamples,
    witness_compute_bootstrap_statistics,
    witness_calculate_bootstrap_intervals,
)

@register_atom(witness_generate_bootstrap_resamples, name="generate_bootstrap_resamples")
@icontract.require(lambda data, n_resamples, seed: data.ndim == 1, "Precondition failed: data.ndim == 1")
@icontract.require(lambda data, n_resamples, seed: n_resamples > 0, "Precondition failed: n_resamples > 0")
@icontract.ensure(lambda result, data, n_resamples, seed: resamples.shape == (n_resamples, len(data)), "Postcondition failed: resamples.shape == (n_resamples, len(data))")
def generate_bootstrap_resamples(data: NDArray[np.float64], n_resamples: int, seed: int = None) -> NDArray[np.float64]:
    """Draw B resamples of size n with replacement from raw data.

    Args:
        data: NDArray[np.float64]
        n_resamples: int
        seed: int | None

    Returns:
        resamples: NDArray[np.float64]
    """
    import numpy.random.Generator
    return numpy.random.Generator.choice(data=data, n_resamples=n_resamples, seed=seed) # type: ignore

@register_atom(witness_compute_bootstrap_statistics, name="compute_bootstrap_statistics")
@icontract.require(lambda resamples, statistic_fn: resamples.ndim == 2, "Precondition failed: resamples.ndim == 2")
@icontract.ensure(lambda result, resamples, statistic_fn: len(result) == resamples.shape[0], "Postcondition failed: len(result) == resamples.shape[0]")
def compute_bootstrap_statistics(resamples: NDArray[np.float64], statistic_fn: Callable[[NDArray[np.float64]], float]) -> NDArray[np.float64]:
    """Map the scalar/vector statistic function over resampled matrices.

    Args:
        resamples: NDArray[np.float64]
        statistic_fn: Callable[[NDArray[np.float64]], float]

    Returns:
        bootstrap_distribution: NDArray[np.float64]
    """
    import numpy
    return numpy.apply_along_axis(resamples=resamples, statistic_fn=statistic_fn) # type: ignore

@register_atom(witness_calculate_bootstrap_intervals, name="calculate_bootstrap_intervals")
@icontract.require(lambda bootstrap_distribution, observed_statistic, confidence_level, method: 0.0 < confidence_level < 1.0, "Precondition failed: 0.0 < confidence_level < 1.0")
@icontract.require(lambda bootstrap_distribution, observed_statistic, confidence_level, method: bootstrap_distribution.ndim == 1, "Precondition failed: bootstrap_distribution.ndim == 1")
@icontract.ensure(lambda result, bootstrap_distribution, observed_statistic, confidence_level, method: result[0] <= result[1], "Postcondition failed: result[0] <= result[1]")
def calculate_bootstrap_intervals(bootstrap_distribution: NDArray[np.float64], observed_statistic: float, confidence_level: float, method: str = None) -> float:
    """Estimate percentile, basic, or BCa boundaries on the resampled distribution.

    Args:
        bootstrap_distribution: NDArray[np.float64]
        observed_statistic: float
        confidence_level: float
        method: str

    Returns:
        ci_bounds: tuple[float, float]
    """
    import scipy.stats
    return scipy.stats.bootstrap(bootstrap_distribution=bootstrap_distribution, observed_statistic=observed_statistic, confidence_level=confidence_level, method=method) # type: ignore

