"""Conditional distribution statistics for additive noise model diagnostics.

Under the additive noise model Y = f(X) + N, if X causes Y, the noise
distribution (and hence its statistical properties) should be approximately
constant across values of X. These atoms discretize X into bins, compute a
statistic of Y within each bin, and return the standard deviation of those
per-bin statistics. High variance signals the wrong causal direction.

Each atom implements the same principle measured via a different statistic:
entropy, skewness, kurtosis, and full distribution shape. The asymmetry
f(X→Y) - f(Y→X) provides evidence for causal direction.

Derived from the 2nd-place solution to the Kaggle Cause-Effect Pairs
challenge (Fonollosa, 2013).

Source: https://github.com/jarfo/cause-effect (Apache 2.0)
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis, skew

import icontract
from sciona.ghost.registry import register_atom

from sciona.atoms.causal_inference.feature_primitives.atoms import (
    CATEGORICAL,
    VarType,
    _count_unique,
    _discrete_entropy,
    _discretized_values,
    _normalize,
    _numerical,
    discretize_and_bin,
)

from .witnesses import (
    witness_conditional_distribution_similarity,
    witness_conditional_noise_entropy_variance,
    witness_conditional_noise_kurtosis_variance,
    witness_conditional_noise_skewness_variance,
)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _discretize_pair(
    x: NDArray[np.float64],
    tx: VarType,
    y: NDArray[np.float64],
    ty: VarType,
    ffactor: int = 3,
    maxdev: int = 3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Discretize both variables of a pair using shared binning parameters."""
    return (
        discretize_and_bin(x, tx, ffactor, maxdev),
        discretize_and_bin(y, ty, ffactor, maxdev),
    )


# ---------------------------------------------------------------------------
# Public atoms
# ---------------------------------------------------------------------------


@register_atom(witness_conditional_noise_entropy_variance)
@icontract.require(lambda x, y: len(x) == len(y), "x and y must have equal length")
@icontract.require(lambda x: len(x) >= 2, "need at least 2 samples")
@icontract.ensure(lambda result: np.isfinite(result) and result >= 0.0, "variance must be non-negative and finite")
def conditional_noise_entropy_variance(
    x: NDArray[np.float64],
    tx: VarType,
    y: NDArray[np.float64],
    ty: VarType,
    ffactor: int = 3,
    maxdev: int = 3,
    minc: int = 10,
) -> float:
    """Compute the standard deviation of conditional entropy of Y|X=x across bins.

    Discretizes X, computes the discrete entropy of Y within each bin of X
    (where the bin has at least minc samples), and returns the std of those
    entropies normalized by log(n_bins_y). Under the additive noise model,
    if X causes Y, the conditional entropy should be approximately constant.

    Args:
        x: Cause candidate variable, shape (n,).
        tx: Type descriptor for x.
        y: Effect candidate variable, shape (n,).
        ty: Type descriptor for y.
        ffactor: Bin granularity factor.
        maxdev: Maximum standard deviations for clipping.
        minc: Minimum samples per bin to include.

    Returns:
        Std of conditional entropies, normalized. Lower in the causal direction.
    """
    xd, yd = _discretize_pair(x, tx, y, ty, ffactor, maxdev)
    cx = Counter(xd.tolist())
    entropies: list[float] = []
    for a in cx:
        if cx[a] > minc:
            entropies.append(_discrete_entropy(y[xd == a], CATEGORICAL))
    if len(entropies) == 0:
        return 0.0
    n = len(_discretized_values(y, ty, ffactor, maxdev))
    if n <= 1:
        return 0.0
    return float(np.std(entropies) / np.log(n))


@register_atom(witness_conditional_noise_skewness_variance)
@icontract.require(lambda x, y: len(x) == len(y), "x and y must have equal length")
@icontract.require(lambda x: len(x) >= 2, "need at least 2 samples")
@icontract.ensure(lambda result: np.isfinite(result) and result >= 0.0, "variance must be non-negative and finite")
def conditional_noise_skewness_variance(
    x: NDArray[np.float64],
    tx: VarType,
    y: NDArray[np.float64],
    ty: VarType,
    ffactor: int = 3,
    maxdev: int = 3,
    minc: int = 8,
) -> float:
    """Compute the standard deviation of conditional skewness of Y|X=x across bins.

    Skewness variation is a higher-order diagnostic than entropy variation.
    It detects asymmetries in the conditional noise distribution that entropy
    (a symmetric measure) may miss, particularly in non-Gaussian settings.

    Args:
        x: Cause candidate variable, shape (n,).
        tx: Type descriptor for x.
        y: Effect candidate variable, shape (n,).
        ty: Type descriptor for y.
        ffactor: Bin granularity factor.
        maxdev: Maximum standard deviations for clipping.
        minc: Minimum samples per bin to include.

    Returns:
        Std of conditional skewness values. Lower in the causal direction.
    """
    xd, _yd = _discretize_pair(x, tx, y, ty, ffactor, maxdev)
    cx = Counter(xd.tolist())
    skewness_values: list[float] = []
    for a in cx:
        if cx[a] >= minc:
            yn = _normalize(y[xd == a], ty)
            skewness_values.append(float(skew(yn)))
    if len(skewness_values) == 0:
        return 0.0
    return float(np.std(skewness_values))


@register_atom(witness_conditional_noise_kurtosis_variance)
@icontract.require(lambda x, y: len(x) == len(y), "x and y must have equal length")
@icontract.require(lambda x: len(x) >= 2, "need at least 2 samples")
@icontract.ensure(lambda result: np.isfinite(result) and result >= 0.0, "variance must be non-negative and finite")
def conditional_noise_kurtosis_variance(
    x: NDArray[np.float64],
    tx: VarType,
    y: NDArray[np.float64],
    ty: VarType,
    ffactor: int = 3,
    maxdev: int = 3,
    minc: int = 8,
) -> float:
    """Compute the standard deviation of conditional kurtosis of Y|X=x across bins.

    Kurtosis captures fourth-moment structure of the noise distribution. In
    the causal direction, kurtosis should be stable (constant noise). In the
    anti-causal direction, the mixing of signal and noise produces varying
    tail behavior.

    Args:
        x: Cause candidate variable, shape (n,).
        tx: Type descriptor for x.
        y: Effect candidate variable, shape (n,).
        ty: Type descriptor for y.
        ffactor: Bin granularity factor.
        maxdev: Maximum standard deviations for clipping.
        minc: Minimum samples per bin to include.

    Returns:
        Std of conditional kurtosis values. Lower in the causal direction.
    """
    xd, _yd = _discretize_pair(x, tx, y, ty, ffactor, maxdev)
    cx = Counter(xd.tolist())
    kurtosis_values: list[float] = []
    for a in cx:
        if cx[a] >= minc:
            yn = _normalize(y[xd == a], ty)
            kurtosis_values.append(float(kurtosis(yn)))
    if len(kurtosis_values) == 0:
        return 0.0
    return float(np.std(kurtosis_values))


@register_atom(witness_conditional_distribution_similarity)
@icontract.require(lambda x, y: len(x) == len(y), "x and y must have equal length")
@icontract.require(lambda x: len(x) >= 2, "need at least 2 samples")
@icontract.ensure(lambda result: np.isfinite(result) and result >= 0.0, "similarity must be non-negative and finite")
def conditional_distribution_similarity(
    x: NDArray[np.float64],
    tx: VarType,
    y: NDArray[np.float64],
    ty: VarType,
    ffactor: int = 2,
    maxdev: int = 3,
    minc: int = 12,
) -> float:
    """Compute shape variation of conditional distributions P(Y|X=x) across bins.

    For each bin of X, computes the conditional distribution P(Y|X=x) as a
    histogram, mean-centers them, and returns the overall standard deviation.
    For numerical data with few unique values, uses cross-correlation
    alignment to detrend the location shift (the deterministic function f(x))
    before comparing shapes.

    This goes beyond moment-based comparisons by capturing the full
    distributional shape variation.

    Args:
        x: Cause candidate variable, shape (n,).
        tx: Type descriptor for x.
        y: Effect candidate variable, shape (n,).
        ty: Type descriptor for y.
        ffactor: Bin granularity factor (default 2 for this atom).
        maxdev: Maximum standard deviations for clipping.
        minc: Minimum samples per bin to include.

    Returns:
        Std of mean-centered conditional distributions. Lower in the causal
        direction.
    """
    xd, yd = _discretize_pair(x, tx, y, ty, ffactor, maxdev)
    cx = Counter(xd.tolist())
    cy = Counter(yd.tolist())
    yrange = sorted(cy.keys())
    ny = len(yrange)
    if ny == 0:
        return 0.0
    py = np.array([cy[i] for i in yrange], dtype=np.float64)
    py = py / py.sum()

    pyx_list: list[NDArray[np.float64]] = []
    for a in cx:
        if cx[a] <= minc:
            continue
        ya = y[xd == a]
        if not _numerical(ty):
            cyx = Counter(ya.tolist())
            pyxa = np.array([cyx.get(i, 0) for i in yrange], dtype=np.float64)
            pyxa.sort()
        elif _count_unique(y) > len(_discretized_values(y, ty, ffactor, maxdev)):
            std_y = np.std(y)
            if std_y > 0:
                ya_norm = (ya - np.mean(ya)) / std_y
            else:
                ya_norm = ya - np.mean(ya)
            ya_binned = discretize_and_bin(ya_norm, ty, ffactor, maxdev, norm=False)
            cyx = Counter(ya_binned.astype(int).tolist())
            dv = _discretized_values(y, ty, ffactor, maxdev)
            pyxa = np.array([cyx.get(int(i), 0) for i in dv], dtype=np.float64)
        else:
            cyx = Counter(ya.tolist())
            pyxa_list = [cyx.get(i, 0) for i in yrange]
            pyxa_padded = np.array(
                [0] * (ny - 1) + pyxa_list + [0] * (ny - 1), dtype=np.float64
            )
            xcorr = [float(np.sum(py * pyxa_padded[i : i + ny])) for i in range(2 * ny - 1)]
            imax = xcorr.index(max(xcorr))
            pyxa = np.array(
                [0] * (2 * ny - 2 - imax) + pyxa_list + [0] * imax, dtype=np.float64
            )
        total = pyxa.sum()
        if total > 0:
            pyxa = pyxa / total
        pyx_list.append(pyxa)

    if len(pyx_list) == 0:
        return 0.0
    pyx = np.array(pyx_list)
    pyx = pyx - pyx.mean(axis=0)
    return float(np.std(pyx))
