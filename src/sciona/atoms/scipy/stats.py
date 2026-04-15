from __future__ import annotations
from typing import Union
import numpy as np
import scipy.stats
import icontract
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from scipy.stats._stats_py import DescribeResult, PearsonRResult, SignificanceResult, TtestResult
from sciona.ghost.registry import register_atom
from sciona.atoms.scipy.witnesses import (
    witness_scipy_describe,
    witness_scipy_norm,
    witness_scipy_pearsonr,
    witness_scipy_spearmanr,
    witness_scipy_ttest_ind,
)

# Types
ArrayLike = Union[np.ndarray, list, tuple]

@register_atom(witness_scipy_describe, name="scipy.stats.describe")
@icontract.require(lambda a: len(np.asarray(a)) > 0, "Input data must not be empty")
@icontract.require(lambda a: np.asarray(a).ndim >= 1, "Input data must be at least 1D")
@icontract.require(lambda a: a is not None, "Input data must not be None")
@icontract.ensure(lambda result: result is not None, "Description result must not be None")
def describe(
    a: ArrayLike,
    axis: int | None = 0,
    ddof: int = 1,
    bias: bool = True,
    nan_policy: str = "propagate",
) -> DescribeResult:
    """Compute several descriptive statistics of the passed array.

    Args:
        a: Input data.
        axis: Axis along which statistics are calculated. Default is 0.
        ddof: Degrees of freedom adjustment for variance.
        bias: If False, skewness (measure of distribution asymmetry)
            and kurtosis (measure of distribution tail heaviness) are
            corrected for statistical bias.
        nan_policy: Defines how to handle input NaNs.

    Returns:
        DescribeResult object containing nobs, minmax, mean, variance,
        skewness (measure of distribution asymmetry), and kurtosis
        (measure of distribution tail heaviness).

    """
    return scipy.stats.describe(
        a,
        axis=axis,
        ddof=ddof,
        bias=bias,
        nan_policy=nan_policy,
    )

@register_atom(witness_scipy_ttest_ind, name="scipy.stats.ttest_ind")
@icontract.require(lambda a: np.asarray(a).ndim >= 1, "Input sample a must be at least 1D")
@icontract.require(lambda b: np.asarray(b).ndim >= 1, "Input sample b must be at least 1D")
@icontract.require(lambda a, b: a is not None and b is not None, "Input samples must not be None")
@icontract.ensure(lambda result: hasattr(result, 'statistic') and hasattr(result, 'pvalue'), "Result must have statistic and pvalue")
def ttest_ind(
    a: ArrayLike,
    b: ArrayLike,
    axis: int = 0,
    equal_var: bool = True,
    nan_policy: str = "propagate",
    alternative: str = "two-sided",
    trim: float = 0,
) -> TtestResult:
    """Calculate the T-test for the means of two independent samples of
    scores.

    Args:
        a, b: The arrays must have the same shape, except in the
            dimension corresponding to axis.
        axis: Axis along which to compute test.
        equal_var: If True, perform a standard independent 2 sample
            test that assumes equal population variances.
        nan_policy: Defines how to handle input NaNs.
        alternative: Defines the alternative hypothesis.
        trim: If non-zero, performs a trimmed (Yuen's) t-test.

    Returns:
        Ttest_indResult object with statistic and pvalue.

    """
    return scipy.stats.ttest_ind(
        a,
        b,
        axis=axis,
        equal_var=equal_var,
        nan_policy=nan_policy,
        alternative=alternative,
        trim=trim,
    )

@register_atom(witness_scipy_pearsonr, name="scipy.stats.pearsonr")
@icontract.require(lambda x, y: len(np.asarray(x)) == len(np.asarray(y)), "x and y must have the same length")
@icontract.require(lambda x: len(np.asarray(x)) >= 2, "Need at least two observations")
@icontract.ensure(lambda result: -1 <= result[0] <= 1, "Correlation coefficient must be between -1 and 1")
def pearsonr(x: ArrayLike, y: ArrayLike) -> PearsonRResult:
    """Pearson correlation coefficient and p-value for testing
    non-correlation.

    Args:
        x: Input array.
        y: Input array.

    Returns:
        PearsonRResult object with statistic and pvalue.

    """
    return scipy.stats.pearsonr(x, y)

@register_atom(witness_scipy_spearmanr, name="scipy.stats.spearmanr")
@icontract.require(lambda a, b: len(np.asarray(a)) == len(np.asarray(b)), "a and b must have the same length")
@icontract.ensure(lambda result: -1 <= result[0] <= 1, "Correlation coefficient must be between -1 and 1")
def spearmanr(
    a: ArrayLike,
    b: ArrayLike | None = None,
    axis: int | None = 0,
    nan_policy: str = "propagate",
    alternative: str = "two-sided",
) -> SignificanceResult:
    """Calculate a Spearman correlation coefficient with associated
    p-value.

    Args:
        a, b: Two 1-D or 2-D arrays containing samples.
        axis: If axis=0 (default), then each column represents a
            variable.
        nan_policy: Defines how to handle input NaNs.
        alternative: Defines the alternative hypothesis.

    Returns:
        SignificanceResult object with statistic and pvalue.

    """
    return scipy.stats.spearmanr(
        a,
        b=b,
        axis=axis,
        nan_policy=nan_policy,
        alternative=alternative,
    )

@register_atom(witness_scipy_norm, name="scipy.stats.norm")
@icontract.require(lambda loc, scale: scale > 0, "Scale must be positive")
@icontract.ensure(lambda result: result is not None, "Normal distribution object must not be None")
def norm(loc: float = 0, scale: float = 1) -> rv_continuous_frozen:
    """A normal continuous random variable.

    Args:
        loc: Mean ("centre") of the distribution.
        scale: Standard deviation of the distribution.

    Returns:
        A frozen normal distribution object.

    """
    return scipy.stats.norm(loc=loc, scale=scale)
