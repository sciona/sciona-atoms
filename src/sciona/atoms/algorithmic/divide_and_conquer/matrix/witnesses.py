"""Ghost witnesses for divide-and-conquer matrix atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_split_matrix_quadrants(
    matrix: AbstractArray,
) -> tuple[AbstractArray, AbstractArray, AbstractArray, AbstractArray]:
    """Ghost witness for split_matrix_quadrants."""
    n = matrix.shape[0] if len(matrix.shape) > 0 else 0
    m = matrix.shape[1] if len(matrix.shape) > 1 else 0
    half_shape = (n // 2, m // 2)
    return (
        AbstractArray(shape=half_shape, dtype=matrix.dtype),
        AbstractArray(shape=half_shape, dtype=matrix.dtype),
        AbstractArray(shape=half_shape, dtype=matrix.dtype),
        AbstractArray(shape=half_shape, dtype=matrix.dtype),
    )


def witness_combine_matrix_quadrants(
    q11: AbstractArray,
    q12: AbstractArray,
    q21: AbstractArray,
    q22: AbstractArray,
) -> AbstractArray:
    """Ghost witness for combine_matrix_quadrants."""
    _ = (q12, q21, q22)
    n = q11.shape[0] if len(q11.shape) > 0 else 0
    m = q11.shape[1] if len(q11.shape) > 1 else 0
    full_shape = (n * 2, m * 2)
    return AbstractArray(shape=full_shape, dtype=q11.dtype)
