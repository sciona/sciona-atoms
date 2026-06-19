from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_cdf_regression_head(
    x: AbstractArray,
    y: AbstractArray,
    distribution: str = "normal",
) -> AbstractArray:
    """Ghost witness for cdf_regression_head."""
    _ = (x, y, distribution)
    # Return 1D array aligned with y
    shape = (y.shape[0],) if hasattr(y, "shape") and len(y.shape) > 0 else (1,)
    return AbstractArray(shape=shape, dtype="float64", min_val=0.0, max_val=1.0)


def witness_mlp_regression_head(
    x: AbstractArray,
    weights: list[AbstractArray],
    biases: list[AbstractArray],
    activation: str = "relu",
) -> AbstractArray:
    """Ghost witness for mlp_regression_head."""
    _ = (x, weights, biases, activation)
    n_samples = x.shape[0] if hasattr(x, "shape") and len(x.shape) > 0 else 1
    out_dim = weights[-1].shape[1] if weights and hasattr(weights[-1], "shape") and len(weights[-1].shape) > 1 else 1
    return AbstractArray(shape=(n_samples, out_dim), dtype="float64")
