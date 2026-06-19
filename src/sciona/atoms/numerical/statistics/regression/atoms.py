from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray
import icontract
from scipy.special import erf
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cdf_regression_head,
    witness_mlp_regression_head,
)


def _mlp_inputs_valid(
    x: NDArray[np.float64],
    weights: list[NDArray[np.float64]],
    biases: list[NDArray[np.float64]],
    activation: str,
) -> bool:
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        return False
    if not isinstance(weights, list) or not isinstance(biases, list):
        return False
    if len(weights) != len(biases) or len(weights) == 0:
        return False
    if activation not in {"relu", "gelu", "sigmoid", "tanh", "identity"}:
        return False

    current_dim = x.shape[1]
    for w, b in zip(weights, biases):
        if not isinstance(w, np.ndarray) or w.ndim != 2:
            return False
        if not isinstance(b, np.ndarray) or b.ndim != 1:
            return False
        if w.shape[0] != current_dim:
            return False
        if w.shape[1] != b.shape[0]:
            return False
        current_dim = w.shape[1]
    return True


def _activate(val: NDArray[np.float64], activation: str) -> NDArray[np.float64]:
    if activation == "relu":
        return np.maximum(0.0, val)
    elif activation == "gelu":
        return val * 0.5 * (1.0 + erf(val / math.sqrt(2.0)))
    elif activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(val, -500.0, 500.0)))
    elif activation == "tanh":
        return np.tanh(val)
    elif activation == "identity":
        return val
    else:
        raise ValueError(f"Unknown activation: {activation}")


@register_atom(witness_cdf_regression_head, name="cdf_regression_head")
@icontract.require(lambda x: isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == 2, "x must be a 2D array of shape (N, 2)")
@icontract.require(lambda y: isinstance(y, np.ndarray) and y.ndim == 1, "y must be a 1D array")
@icontract.require(lambda x, y: x.shape[0] == y.shape[0], "x and y must have the same length")
@icontract.require(lambda distribution: distribution in {"normal", "logistic"}, "distribution must be normal or logistic")
@icontract.ensure(lambda result: isinstance(result, np.ndarray) and result.ndim == 1, "result must be a 1D array")
@icontract.ensure(lambda result, y: result.shape[0] == y.shape[0], "result must have the same length as y")
@icontract.ensure(lambda result: np.all((result >= 0.0) & (result <= 1.0)), "CDF values must be in [0, 1]")
def cdf_regression_head(x: NDArray[np.float64], y: NDArray[np.float64], distribution: str = "normal") -> NDArray[np.float64]:
    """Compute the cumulative probability for target values given predicted parameters.

    Args:
        x: predictions of shape (N, 2) where x[:, 0] is mean/location and x[:, 1] is log-scale.
        y: target values of shape (N,).
        distribution: the target CDF distribution ('normal' or 'logistic').
    """
    loc = x[:, 0]
    scale = np.exp(x[:, 1])

    if distribution == "normal":
        z = (y - loc) / (scale * math.sqrt(2.0))
        return 0.5 * (1.0 + erf(z))
    elif distribution == "logistic":
        z = (y - loc) / scale
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


@register_atom(witness_mlp_regression_head, name="mlp_regression_head")
@icontract.require(
    lambda x, weights, biases, activation: _mlp_inputs_valid(x, weights, biases, activation),
    "MLP parameters must have matching shapes and a supported activation function",
)
@icontract.ensure(
    lambda result, x, weights: isinstance(result, np.ndarray)
    and result.ndim == 2
    and result.shape[0] == x.shape[0]
    and result.shape[1] == weights[-1].shape[1],
    "Output must have shape (N, D_out)",
)
def mlp_regression_head(
    x: NDArray[np.float64],
    weights: list[NDArray[np.float64]],
    biases: list[NDArray[np.float64]],
    activation: str = "relu",
) -> NDArray[np.float64]:
    """Forward pass of an MLP regression head.

    Args:
        x: input features of shape (N, D_in).
        weights: list of weights for each layer.
        biases: list of bias vectors.
        activation: activation function name to apply to hidden layers.
    """
    current = x
    n_layers = len(weights)
    for i in range(n_layers):
        current = np.dot(current, weights[i]) + biases[i]
        if i < n_layers - 1:
            current = _activate(current, activation)
    return current
