"""Causal inference feature primitives for pairwise causal direction detection.

Implements statistical tests and information-theoretic measures used to
determine causal direction between variable pairs. Derived from the 2nd-place
solution to the Kaggle Cause-Effect Pairs challenge (Fonollosa, 2013).

Each atom computes an asymmetric score: the difference between f(X→Y) and
f(Y→X) provides evidence for the causal direction. The atoms cover distinct
theoretical frameworks: information-geometric causal inference (IGCI),
kernel independence testing (HSIC), additive noise model diagnostics,
Bayesian error rates, and distributional complexity measures.

Source: https://github.com/jarfo/cause-effect (Apache 2.0)
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from numpy.typing import NDArray
from scipy.special import psi

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_discretize_and_bin,
    witness_hsic_independence_test,
    witness_igci_asymmetry_score,
    witness_knn_entropy_estimator,
    witness_normalized_error_probability,
    witness_polyfit_nonlinearity_asymmetry,
    witness_polyfit_residual_error,
    witness_uniform_divergence,
)

# ---------------------------------------------------------------------------
# Variable type constants
# ---------------------------------------------------------------------------

BINARY: str = "Binary"
CATEGORICAL: str = "Categorical"
NUMERICAL: str = "Numerical"

VarType = str  # One of BINARY, CATEGORICAL, NUMERICAL

# ---------------------------------------------------------------------------
# Shared helpers (private — not atoms, not exported)
# ---------------------------------------------------------------------------


def _numerical(tp: VarType) -> bool:
    return tp == NUMERICAL


def _categorical(tp: VarType) -> bool:
    return tp == CATEGORICAL


def _count_unique(x: NDArray[np.float64]) -> int:
    return len(set(x.tolist()))


def _normalize(x: NDArray[np.float64], tx: VarType) -> NDArray[np.float64]:
    """Normalize variable: frequency-rank for categorical, z-score for numerical."""
    if not _numerical(tx):
        cx = Counter(x.tolist())
        xmap: dict[float, int] = {}
        for i, (k, _) in enumerate(cx.most_common()):
            xmap[k] = i
        y = np.array([xmap[v] for v in x.tolist()], dtype=np.float64)
    else:
        y = x.copy()
    y = y - np.mean(y)
    std = np.std(y)
    if std > 0:
        y = y / std
    return y


def _to_numerical(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convert categorical x to numerical by mean-of-y encoding."""
    dx: dict[float, NDArray[np.float64]] = defaultdict(lambda: np.zeros(2))
    for i, a in enumerate(x.tolist()):
        dx[a][0] += y[i]
        dx[a][1] += 1
    for a in dx:
        dx[a][0] /= dx[a][1]
    return np.array([dx[a][0] for a in x.tolist()], dtype=np.float64)


def _discretized_values(
    x: NDArray[np.float64], tx: VarType, ffactor: int, maxdev: int
) -> list[float]:
    if _numerical(tx) and _count_unique(x) > (2 * ffactor * maxdev + 1):
        vmax = ffactor * maxdev
        vmin = -ffactor * maxdev
        return list(range(vmin, vmax + 1))
    return sorted(set(x.tolist()))


def _discrete_probability(
    x: NDArray[np.float64], tx: VarType, ffactor: int, maxdev: int
) -> Counter[float]:
    xd = discretize_and_bin(x, tx, ffactor, maxdev)
    return Counter(xd.tolist())


def _discrete_entropy(
    x: NDArray[np.float64],
    tx: VarType,
    ffactor: int = 3,
    maxdev: int = 3,
    bias_factor: float = 0.7,
) -> float:
    c = _discrete_probability(x, tx, ffactor, maxdev)
    pk = np.array(list(c.values()), dtype=np.float64)
    pk = pk / pk.sum()
    valid = pk > 0
    entropy = -float(np.sum(pk[valid] * np.log(pk[valid])))
    n = len(pk)
    if n > 1:
        entropy += bias_factor * (n - 1) / (2.0 * len(x))
    return entropy


# ---------------------------------------------------------------------------
# Public atoms
# ---------------------------------------------------------------------------


@register_atom(witness_igci_asymmetry_score)
@icontract.require(lambda x, y: len(x) == len(y), "x and y must have equal length")
@icontract.require(lambda x: len(x) >= 2, "need at least 2 samples")
@icontract.ensure(lambda result: np.isfinite(result), "score must be finite")
def igci_asymmetry_score(
    x: NDArray[np.float64],
    tx: VarType,
    y: NDArray[np.float64],
    ty: VarType,
) -> float:
    """Compute the Information-Geometric Causal Inference (IGCI) asymmetry score.

    Sorts samples by x, computes the log-ratio of consecutive x-deltas to
    y-deltas. Under the IGCI framework (Daniusis et al., 2012), if X causes
    Y via a deterministic function, the distribution of the slope (Jacobian)
    is independent of the cause distribution. The score is asymmetric:
    igci(X, Y) != igci(Y, X), and the difference indicates causal direction.

    Handles non-injective mappings by averaging y values per unique x, and
    weights log-ratios by the minimum count of adjacent bins.

    Args:
        x: Cause candidate variable, shape (n,).
        tx: Type of x (BINARY, CATEGORICAL, or NUMERICAL).
        y: Effect candidate variable, shape (n,).
        ty: Type of y.

    Returns:
        IGCI asymmetry score. Compute igci(X,Y) - igci(Y,X) for direction.
    """
    if _count_unique(x) < 2:
        return 0.0
    xn = _normalize(x, tx)
    yn = _normalize(y, ty)
    if len(xn) != _count_unique(xn):
        dx: dict[float, NDArray[np.float64]] = defaultdict(lambda: np.zeros(2))
        for i, a in enumerate(xn.tolist()):
            dx[a][0] += yn[i]
            dx[a][1] += 1
        for a in dx:
            dx[a][0] /= dx[a][1]
        xy = np.array(sorted([[a, dx[a][0]] for a in dx]), dtype=np.float64)
        counter = np.array([dx[a][1] for a in xy[:, 0].tolist()], dtype=np.float64)
    else:
        order = np.argsort(xn)
        xy = np.column_stack([xn[order], yn[order]])
        counter = np.ones(len(xn))
    delta = xy[1:] - xy[:-1]
    selec = delta[:, 1] != 0
    delta = delta[selec]
    counter_min = np.minimum(counter[1:], counter[:-1])[selec]
    return float(np.sum(counter_min * np.log(delta[:, 0] / np.abs(delta[:, 1]))) / len(x))


@register_atom(witness_hsic_independence_test)
@icontract.require(lambda X, Y: len(X) == len(Y), "X and Y must have equal length")
@icontract.require(lambda X: len(X) >= 2, "need at least 2 samples")
@icontract.ensure(lambda result: np.isfinite(result), "statistic must be finite")
def hsic_independence_test(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    sig: list[float] | None = None,
    maxpnt: int = 200,
) -> float:
    """Compute the Hilbert-Schmidt Independence Criterion (HSIC) test statistic.

    Measures statistical dependence between X and Y using RBF kernels with
    automatic median-distance bandwidth selection (Gretton et al., 2005).
    HSIC = (1/m) * trace(K_c @ L_c) where K_c, L_c are centered kernel
    matrices.

    Uses evenly-spaced subsampling (not random) when n > maxpnt to preserve
    distribution shape while keeping the O(n^2) kernel computation tractable.

    Args:
        X: First variable, shape (n,) or (n, d).
        Y: Second variable, shape (n,) or (n, d).
        sig: Kernel bandwidths [sig_x, sig_y]. Use -1 for median heuristic.
        maxpnt: Maximum points for subsampling.

    Returns:
        HSIC test statistic. Higher values indicate stronger dependence.
    """
    if sig is None:
        sig = [-1.0, -1.0]

    m = X.shape[0]
    if m > maxpnt:
        indx = np.floor(np.linspace(0, m - 1, maxpnt)).astype(int)
        Xm = X[indx].astype(np.float64)
        Ym = Y[indx].astype(np.float64)
        m = Xm.shape[0]
    else:
        Xm = X.astype(np.float64)
        Ym = Y.astype(np.float64)

    H = np.eye(m) - (1.0 / m) * np.ones((m, m))

    K = _rbf_dot(Xm, sig[0])
    L = _rbf_dot(Ym, sig[1])

    Kc = H @ K @ H
    Lc = H @ L @ H

    stat = float((1.0 / m) * np.sum(Kc.T * Lc))
    return stat if np.isfinite(stat) else 0.0


def _rbf_dot(X: NDArray[np.float64], deg: float) -> NDArray[np.float64]:
    """RBF kernel matrix with optional median-distance bandwidth."""
    if X.ndim == 1:
        X = X[:, np.newaxis]
    m = X.shape[0]
    G = np.sum(X * X, axis=1)[:, np.newaxis]
    Q = np.tile(G, (1, m))
    H = Q + Q.T - 2.0 * np.dot(X, X.T)
    if deg == -1:
        dists = (H - np.tril(H)).flatten()
        positive = dists[dists > 0]
        deg = np.sqrt(0.5 * np.median(positive)) if len(positive) > 0 else 1.0
    return np.exp(-H / (2.0 * deg ** 2))


@register_atom(witness_knn_entropy_estimator)
@icontract.require(lambda x: len(x) >= 2, "need at least 2 samples")
@icontract.ensure(lambda result: np.isfinite(result), "entropy must be finite")
def knn_entropy_estimator(
    x: NDArray[np.float64],
    tx: VarType,
    m: int = 2,
) -> float:
    """Estimate differential entropy using a k-nearest-neighbor approach.

    Kozachenko-Leonenko type estimator adapted for mixed discrete/continuous
    data. Groups identical values, computes inter-point spacings weighted by
    counts, and applies digamma bias corrections.

    Args:
        x: Variable samples, shape (n,).
        tx: Variable type.
        m: KNN neighbor count for spacing estimation.

    Returns:
        Estimated differential entropy (nats).
    """
    xn = _normalize(x, tx)
    cx = Counter(xn.tolist())
    if len(cx) < 2:
        return 0.0
    xk = np.array(sorted(cx.keys()), dtype=np.float64)
    delta = (xk[1:] - xk[:-1]) / m
    counter = np.array([cx[k] for k in xk.tolist()], dtype=np.float64)
    hx = float(np.sum(counter[1:] * np.log(delta / counter[1:])) / len(x))
    hx += psi(len(delta)) - np.log(len(delta))
    hx += np.log(len(x))
    hx -= psi(m) - np.log(m)
    return hx


@register_atom(witness_uniform_divergence)
@icontract.require(lambda x: len(x) >= 1, "need at least 1 sample")
@icontract.ensure(lambda result: np.isfinite(result), "divergence must be finite")
def uniform_divergence(
    x: NDArray[np.float64],
    tx: VarType,
    m: int = 2,
) -> float:
    """Compute KL-divergence-like measure from the empirical distribution to uniform.

    Under the independence of cause and mechanism principle (Lemeire & Janzing,
    2012), the cause distribution tends to be "simpler" (closer to uniform).
    The asymmetry between uniform_divergence(X) and uniform_divergence(Y)
    indicates which variable is more likely the cause.

    Uses a KNN density estimate with variable-width bins.

    Args:
        x: Variable samples, shape (n,).
        tx: Variable type.
        m: Neighbor count for density estimation.

    Returns:
        Divergence from uniform. Lower values suggest the variable is a cause.
    """
    xn = _normalize(x, tx)
    cx = Counter(xn.tolist())
    xk = np.array(sorted(cx.keys()), dtype=np.float64)
    delta = np.zeros(len(xk))
    if len(xk) > 1:
        delta[0] = xk[1] - xk[0]
        if len(xk) > m:
            delta[1:-1] = (xk[m:] - xk[:-m]) / m
        delta[-1] = xk[-1] - xk[-2]
    else:
        delta = np.array([np.sqrt(12.0)])
    counter = np.array([cx[k] for k in xk.tolist()], dtype=np.float64)
    delta = delta / np.sum(delta)
    hx = float(np.sum(counter * np.log(counter / delta)) / len(x))
    hx -= np.log(len(x))
    hx += psi(m) - np.log(m)
    return hx


@register_atom(witness_normalized_error_probability)
@icontract.require(lambda x, y: len(x) == len(y), "x and y must have equal length")
@icontract.require(lambda x: len(x) >= 2, "need at least 2 samples")
@icontract.ensure(lambda result: 0.0 <= result <= 1.0, "probability must be in [0, 1]")
def normalized_error_probability(
    x: NDArray[np.float64],
    tx: VarType,
    y: NDArray[np.float64],
    ty: VarType,
    ffactor: int = 3,
    maxdev: int = 3,
) -> float:
    """Compute normalized Bayesian error rate of predicting Y from X.

    Discretizes both variables, constructs the joint probability table, and
    computes the MAP error rate normalized by the maximum possible error rate.
    The asymmetry nep(X→Y) != nep(Y→X) provides causal signal because the
    cause typically has lower normalized error in predicting the effect.

    Args:
        x: Predictor variable, shape (n,).
        tx: Type of x.
        y: Target variable, shape (n,).
        ty: Type of y.
        ffactor: Bin granularity factor (bins per standard deviation).
        maxdev: Maximum deviations for clipping.

    Returns:
        Normalized error probability in [0, 1].
    """
    xd = discretize_and_bin(x, tx, ffactor, maxdev)
    yd = discretize_and_bin(y, ty, ffactor, maxdev)
    cx = Counter(xd.tolist())
    cy = Counter(yd.tolist())
    pxy: dict[tuple[float, float], int] = defaultdict(int)
    for xi, yi in zip(xd.tolist(), yd.tolist()):
        pxy[(xi, yi)] += 1
    joint = np.array(
        [[pxy[(a, b)] for b in cy] for a in cx], dtype=np.float64
    )
    joint = joint / joint.sum()
    perr = 1.0 - float(np.sum(joint.max(axis=1)))
    max_perr = 1.0 - float(np.max(joint.sum(axis=0)))
    return perr / max_perr if max_perr > 0 else perr


@register_atom(witness_discretize_and_bin)
@icontract.require(lambda x: len(x) >= 1, "need at least 1 sample")
@icontract.ensure(lambda result: len(result) > 0, "output must be non-empty")
def discretize_and_bin(
    x: NDArray[np.float64],
    tx: VarType,
    ffactor: int = 3,
    maxdev: int = 3,
    norm: bool = True,
) -> NDArray[np.float64]:
    """Discretize a continuous variable into integer bins with robust normalization.

    Two-pass normalization: first pass identifies inliers within maxdev
    standard deviations, second pass re-centers/re-scales using only those
    inliers. This prevents outliers from distorting the binning.

    Args:
        x: Variable samples, shape (n,).
        tx: Variable type.
        ffactor: Bins per standard deviation (default 3 → 19 bins total).
        maxdev: Maximum standard deviations for clipping.
        norm: Whether to apply normalization.

    Returns:
        Discretized values, shape (n,).
    """
    xd = x.copy()
    n_unique = _count_unique(xd)
    n_bins = len(_discretized_values(xd, tx, ffactor, maxdev))
    if not norm or (_numerical(tx) and n_unique > n_bins):
        if norm:
            std = np.std(xd)
            if std > 0:
                xd = (xd - np.mean(xd)) / std
            xf = xd[np.abs(xd) < maxdev]
            std_f = np.std(xf)
            if std_f > 0:
                xd = (xd - np.mean(xf)) / std_f
        xd = np.round(xd * ffactor)
        vmax = float(ffactor * maxdev)
        vmin = float(-ffactor * maxdev)
        xd = np.clip(xd, vmin, vmax)
    return xd


@register_atom(witness_polyfit_nonlinearity_asymmetry)
@icontract.require(lambda x, y: len(x) == len(y), "x and y must have equal length")
@icontract.ensure(lambda result: result >= 0.0, "asymmetry score must be non-negative")
def polyfit_nonlinearity_asymmetry(
    x: NDArray[np.float64],
    tx: VarType,
    y: NDArray[np.float64],
    ty: VarType,
) -> float:
    """Quantify nonlinearity asymmetry between X→Y and Y→X polynomial fits.

    Fits degree-1 and degree-2 polynomials from X to Y, then returns
    |2*a2| + |a2_1 - a1_0| where a2 is the quadratic coefficient and a2_1,
    a1_0 are the linear coefficients from degree-2 and degree-1 fits.

    Causal mechanisms tend to be "simpler" in the forward direction. If
    X→Y is approximately linear but Y→X is nonlinear, the difference
    signals the causal direction.

    Args:
        x: Cause candidate, shape (n,).
        tx: Type of x.
        y: Effect candidate, shape (n,).
        ty: Type of y.

    Returns:
        Nonlinearity score. Compute fit(X,Y) - fit(Y,X) for direction.
    """
    if not _numerical(tx) or not _numerical(ty):
        return 0.0
    if _count_unique(x) <= 2 or _count_unique(y) <= 2:
        return 0.0
    xn = (x - np.mean(x)) / np.std(x)
    yn = (y - np.mean(y)) / np.std(y)
    xy1 = np.polyfit(xn, yn, 1)
    xy2 = np.polyfit(xn, yn, 2)
    return float(abs(2 * xy2[0]) + abs(xy2[1] - xy1[0]))


@register_atom(witness_polyfit_residual_error)
@icontract.require(lambda x, y: len(x) == len(y), "x and y must have equal length")
@icontract.ensure(lambda result: result >= 0.0, "residual error must be non-negative")
def polyfit_residual_error(
    x: NDArray[np.float64],
    tx: VarType,
    y: NDArray[np.float64],
    ty: VarType,
    m: int = 2,
) -> float:
    """Compute polynomial regression residual standard deviation.

    Fits a degree-m polynomial from X to Y and returns the residual std.
    Type-aware: categorical X is converted via mean-of-Y encoding.
    The asymmetry fit_error(X→Y) - fit_error(Y→X) implements a practical
    additive noise model test.

    Args:
        x: Predictor variable, shape (n,).
        tx: Type of x.
        y: Target variable, shape (n,).
        ty: Type of y.
        m: Polynomial degree.

    Returns:
        Standard deviation of residuals.
    """
    xw = x.copy()
    yw = y.copy()
    if _categorical(tx) and _categorical(ty):
        xw = _normalize(xw, tx)
        yw = _normalize(yw, ty)
    elif _categorical(tx) and _numerical(ty):
        xw = _to_numerical(xw, yw)
    elif _numerical(tx) and _categorical(ty):
        yw = _to_numerical(yw, xw)
    std_x = np.std(xw)
    std_y = np.std(yw)
    if std_x > 0:
        xw = (xw - np.mean(xw)) / std_x
    if std_y > 0:
        yw = (yw - np.mean(yw)) / std_y
    deg = m
    if _count_unique(xw) <= m or _count_unique(yw) <= m:
        deg = min(_count_unique(xw), _count_unique(yw)) - 1
    if deg < 1:
        return float(np.std(yw))
    coeffs = np.polyfit(xw, yw, deg)
    return float(np.std(yw - np.polyval(coeffs, xw)))
