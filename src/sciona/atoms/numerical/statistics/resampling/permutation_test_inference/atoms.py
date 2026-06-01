from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_observed_statistic,
    witness_generate_permutation_null,
    witness_calculate_permutation_p_value,
)

@register_atom(witness_compute_observed_statistic, name="compute_observed_statistic")
@icontract.require(lambda x, y, metric: x.ndim == 1, "Precondition failed: x.ndim == 1")
@icontract.require(lambda x, y, metric: y.ndim == 1, "Precondition failed: y.ndim == 1")
@icontract.ensure(lambda result, x, y, metric: result is not None, "Postcondition failed: result is not None")
def compute_observed_statistic(x: NDArray[np.float64], y: NDArray[np.float64], metric: str) -> float:
    """Evaluate the raw metric (difference in mean, median, etc.) on the unshuffled samples.

    Args:
        x: NDArray[np.float64]
        y: NDArray[np.float64]
        metric: str

    Returns:
        observed_stat: float
    """
    import numpy
    return numpy.mean(x=x, y=y, metric=metric) # type: ignore

@register_atom(witness_generate_permutation_null, name="generate_permutation_null")
@icontract.require(lambda x, y, metric, n_permutations, seed: n_permutations > 0, "Precondition failed: n_permutations > 0")
@icontract.ensure(lambda result, x, y, metric, n_permutations, seed: len(result) == n_permutations, "Postcondition failed: len(result) == n_permutations")
def generate_permutation_null(x: NDArray[np.float64], y: NDArray[np.float64], metric: str, n_permutations: int, seed: int = None) -> NDArray[np.float64]:
    """Shuffle grouped vectors B times without replacement, computing the statistic on each shuffle.

    Args:
        x: NDArray[np.float64]
        y: NDArray[np.float64]
        metric: str
        n_permutations: int
        seed: int | None

    Returns:
        null_distribution: NDArray[np.float64]
    """
    import numpy.random.Generator
    return numpy.random.Generator.shuffle(x=x, y=y, metric=metric, n_permutations=n_permutations, seed=seed) # type: ignore

@register_atom(witness_calculate_permutation_p_value, name="calculate_permutation_p_value")
@icontract.require(lambda null_distribution, observed_stat, alternative: alternative in ['two-sided', 'less', 'greater'], "Precondition failed: alternative in ['two-sided', 'less', 'greater']")
@icontract.ensure(lambda result, null_distribution, observed_stat, alternative: 0.0 <= result <= 1.0, "Postcondition failed: 0.0 <= result <= 1.0")
def calculate_permutation_p_value(null_distribution: NDArray[np.float64], observed_stat: float, alternative: str = None) -> float:
    """Evaluate the proportion of the null distribution exceeding the observed statistic.

    Args:
        null_distribution: NDArray[np.float64]
        observed_stat: float
        alternative: str

    Returns:
        p_value: float
    """
    import scipy.stats
    return scipy.stats.permutation_test(null_distribution=null_distribution, observed_stat=observed_stat, alternative=alternative) # type: ignore

