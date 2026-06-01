from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_test_variance_homogeneity,
    witness_compute_t_moments,
    witness_compute_welch_t_statistic,
    witness_evaluate_t_significance,
)

@register_atom(witness_test_variance_homogeneity, name="test_variance_homogeneity")
@icontract.require(lambda x1, x2: len(x1) >= 3, "Precondition failed: len(x1) >= 3")
@icontract.require(lambda x1, x2: len(x2) >= 3, "Precondition failed: len(x2) >= 3")
@icontract.ensure(lambda result, x1, x2: 0.0 <= result <= 1.0, "Postcondition failed: 0.0 <= result <= 1.0")
def test_variance_homogeneity(x1: NDArray[np.float64], x2: NDArray[np.float64]) -> float:
    """Perform Levene's test to evaluate equality of variances between two groups.

    Args:
        x1: NDArray[np.float64]
        x2: NDArray[np.float64]

    Returns:
        levene_p: float
    """
    import scipy.stats
    return scipy.stats.levene(x1=x1, x2=x2) # type: ignore

@register_atom(witness_compute_t_moments, name="compute_t_moments")
@icontract.require(lambda x: x.ndim == 1, "Precondition failed: x.ndim == 1")
@icontract.ensure(lambda result, x: variance >= 0.0, "Postcondition failed: variance >= 0.0")
@icontract.ensure(lambda result, x: n > 0, "Postcondition failed: n > 0")
def compute_t_moments(x: NDArray[np.float64]) -> float:
    """Calculate sample means, standard deviations, and sizes.

    Args:
        x: NDArray[np.float64]

    Returns:
        mean: float
    """
    import numpy
    return numpy.mean(x=x) # type: ignore

@register_atom(witness_compute_welch_t_statistic, name="compute_welch_t_statistic")
@icontract.require(lambda mean1, variance1, n1, mean2, variance2, n2: variance1 > 0.0, "Precondition failed: variance1 > 0.0")
@icontract.require(lambda mean1, variance1, n1, mean2, variance2, n2: variance2 > 0.0, "Precondition failed: variance2 > 0.0")
@icontract.require(lambda mean1, variance1, n1, mean2, variance2, n2: n1 > 1, "Precondition failed: n1 > 1")
@icontract.require(lambda mean1, variance1, n1, mean2, variance2, n2: n2 > 1, "Precondition failed: n2 > 1")
@icontract.ensure(lambda result, mean1, variance1, n1, mean2, variance2, n2: df > 1.0, "Postcondition failed: df > 1.0")
def compute_welch_t_statistic(mean1: float, variance1: float, n1: int, mean2: float, variance2: float, n2: int) -> float:
    """Calculate Welch t-statistic and Welch-Satterthwaite degrees of freedom.

    Args:
        mean1: float
        variance1: float
        n1: int
        mean2: float
        variance2: float
        n2: int

    Returns:
        t_stat: float
    """
    import scipy.stats
    return scipy.stats.ttest_ind(mean1=mean1, variance1=variance1, n1=n1, mean2=mean2, variance2=variance2, n2=n2) # type: ignore

@register_atom(witness_evaluate_t_significance, name="evaluate_t_significance")
@icontract.require(lambda t_stat, df, alternative: df > 0.0, "Precondition failed: df > 0.0")
@icontract.ensure(lambda result, t_stat, df, alternative: 0.0 <= p_value <= 1.0, "Postcondition failed: 0.0 <= p_value <= 1.0")
def evaluate_t_significance(t_stat: float, df: float, alternative: str = None) -> float:
    """Evaluate p-value and confidence interval using student-t cumulative distribution function.

    Args:
        t_stat: float
        df: float
        alternative: str

    Returns:
        p_value: float
    """
    import scipy.stats.t
    return scipy.stats.t.cdf(t_stat=t_stat, df=df, alternative=alternative) # type: ignore

