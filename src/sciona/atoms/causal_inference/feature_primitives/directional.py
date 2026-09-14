"""Directional feature composition with explicit shape and numeric contracts."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_asymmetric_feature_difference(
    features_ab: AbstractArray, features_ba: AbstractArray,
) -> AbstractArray:
    if features_ab.shape != features_ba.shape:
        raise ValueError("directional feature shapes must match")
    return AbstractArray(shape=features_ab.shape, dtype="float64")


@register_atom(witness_asymmetric_feature_difference)
def asymmetric_feature_difference(
    features_ab: NDArray[np.float64],
    features_ba: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return forward minus reverse features, preserving the input shape.

    Inputs must be nonempty, equally shaped, finite real numeric arrays. There
    is no broadcasting or implicit pairing of samples. Subtraction uses float64
    to avoid unsigned wraparound, leaves both inputs unchanged, and rejects
    overflow. Swapping the inputs negates the result. This operation combines
    already computed features; it does not estimate causal direction itself.
    """
    for values in (features_ab, features_ba):
        if not isinstance(values, np.ndarray):
            raise TypeError("directional features must be NumPy arrays")
        if values.ndim == 0 or values.size == 0:
            raise ValueError("directional features must be nonempty arrays")
        if values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
            raise ValueError("directional features must be finite real numbers")
    if features_ab.shape != features_ba.shape:
        raise ValueError("directional feature shapes must match")
    with np.errstate(over="raise", invalid="raise"):
        return np.subtract(features_ab, features_ba, dtype=np.float64)
