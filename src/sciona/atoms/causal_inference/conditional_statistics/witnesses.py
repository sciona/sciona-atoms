"""Ghost witnesses for conditional distribution statistics."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_conditional_noise_entropy_variance(
    x: AbstractArray, tx: str, y: AbstractArray, ty: str,
    ffactor: int = 3, maxdev: int = 3, minc: int = 10,
) -> float:
    """Ghost witness for conditional noise entropy variance.

    Takes two equal-length arrays and binning parameters, returns a
    non-negative scalar float. Shape-invariant.
    """
    return 0.0


def witness_conditional_noise_skewness_variance(
    x: AbstractArray, tx: str, y: AbstractArray, ty: str,
    ffactor: int = 3, maxdev: int = 3, minc: int = 8,
) -> float:
    """Ghost witness for conditional noise skewness variance.

    Takes two equal-length arrays and binning parameters, returns a
    non-negative scalar float. Shape-invariant.
    """
    return 0.0


def witness_conditional_noise_kurtosis_variance(
    x: AbstractArray, tx: str, y: AbstractArray, ty: str,
    ffactor: int = 3, maxdev: int = 3, minc: int = 8,
) -> float:
    """Ghost witness for conditional noise kurtosis variance.

    Takes two equal-length arrays and binning parameters, returns a
    non-negative scalar float. Shape-invariant.
    """
    return 0.0


def witness_conditional_distribution_similarity(
    x: AbstractArray, tx: str, y: AbstractArray, ty: str,
    ffactor: int = 2, maxdev: int = 3, minc: int = 12,
) -> float:
    """Ghost witness for conditional distribution similarity.

    Takes two equal-length arrays and binning parameters, returns a
    non-negative scalar float. Shape-invariant.
    """
    return 0.0
