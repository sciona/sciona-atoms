"""Ghost witnesses for causal inference feature primitives.

Each witness mirrors the atom's interface using abstract types and captures
the semantic shape of the computation without executing it.
"""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_igci_asymmetry_score(
    x: AbstractArray, tx: str, y: AbstractArray, ty: str
) -> float:
    """Ghost witness for IGCI asymmetry score.

    Takes two equal-length arrays and type descriptors, returns a scalar
    float score. Shape-invariant: output does not depend on input shape.
    """
    return 0.0


def witness_hsic_independence_test(
    X: AbstractArray,
    Y: AbstractArray,
    sig: list[float] | None = None,
    maxpnt: int = 200,
) -> float:
    """Ghost witness for HSIC independence test.

    Takes two equal-length arrays, returns a scalar float statistic.
    The subsampling parameter maxpnt does not affect the output shape.
    """
    return 0.0


def witness_knn_entropy_estimator(
    x: AbstractArray, tx: str, m: int = 2
) -> float:
    """Ghost witness for KNN entropy estimator.

    Takes a 1D array and type descriptor, returns a scalar float entropy.
    """
    return 0.0


def witness_uniform_divergence(
    x: AbstractArray, tx: str, m: int = 2
) -> float:
    """Ghost witness for uniform divergence.

    Takes a 1D array and type descriptor, returns a scalar float divergence.
    """
    return 0.0


def witness_normalized_error_probability(
    x: AbstractArray,
    tx: str,
    y: AbstractArray,
    ty: str,
    ffactor: int = 3,
    maxdev: int = 3,
) -> float:
    """Ghost witness for normalized error probability.

    Takes two equal-length arrays, returns a scalar float in [0, 1].
    """
    return 0.0


def witness_discretize_and_bin(
    x: AbstractArray,
    tx: str,
    ffactor: int = 3,
    maxdev: int = 3,
    norm: bool = True,
) -> AbstractArray:
    """Ghost witness for discretize-and-bin.

    Takes a 1D array, returns a 1D array of the same shape with
    discretized (integer-valued) elements.
    """
    return AbstractArray(shape=x.shape, dtype="float64")


def witness_polyfit_nonlinearity_asymmetry(
    x: AbstractArray, tx: str, y: AbstractArray, ty: str
) -> float:
    """Ghost witness for polynomial nonlinearity asymmetry.

    Takes two equal-length arrays, returns a non-negative scalar float.
    """
    return 0.0


def witness_polyfit_residual_error(
    x: AbstractArray,
    tx: str,
    y: AbstractArray,
    ty: str,
    m: int = 2,
) -> float:
    """Ghost witness for polynomial residual error.

    Takes two equal-length arrays, returns a non-negative scalar float.
    """
    return 0.0
