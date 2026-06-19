"""Divide-and-conquer matrix atoms for Strassen's algorithm."""

from __future__ import annotations

import icontract
import numpy as np
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_split_matrix_quadrants,
    witness_combine_matrix_quadrants,
)


@register_atom(witness_split_matrix_quadrants)
@icontract.require(lambda matrix: matrix.ndim == 2, "Matrix must be 2D")
@icontract.require(lambda matrix: matrix.shape[0] % 2 == 0 and matrix.shape[1] % 2 == 0, "Matrix dimensions must be even")
@icontract.ensure(
    lambda result, matrix: (
        result[0].shape == (matrix.shape[0] // 2, matrix.shape[1] // 2) and
        result[1].shape == (matrix.shape[0] // 2, matrix.shape[1] // 2) and
        result[2].shape == (matrix.shape[0] // 2, matrix.shape[1] // 2) and
        result[3].shape == (matrix.shape[0] // 2, matrix.shape[1] // 2)
    ),
    "All output quadrants must be exactly half the shape of the input matrix"
)
def split_matrix_quadrants(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a matrix into four submatrices for recursive multiplication.

    Args:
        matrix: The input 2D numpy array to split.

    Returns:
        A tuple of four numpy arrays representing q11, q12, q21, and q22.
    """
    n, m = matrix.shape
    mid_row = n // 2
    mid_col = m // 2
    q11 = matrix[:mid_row, :mid_col]
    q12 = matrix[:mid_row, mid_col:]
    q21 = matrix[mid_row:, :mid_col]
    q22 = matrix[mid_row:, mid_col:]
    return q11, q12, q21, q22


@register_atom(witness_combine_matrix_quadrants)
@icontract.require(lambda q11, q12, q21, q22: q11.ndim == 2 and q12.ndim == 2 and q21.ndim == 2 and q22.ndim == 2, "All quadrants must be 2D")
@icontract.require(lambda q11, q12, q21, q22: q11.shape == q12.shape == q21.shape == q22.shape, "All quadrants must have the same shape")
@icontract.ensure(lambda result, q11: result.shape == (q11.shape[0] * 2, q11.shape[1] * 2), "Output shape must be twice the size of quadrants")
def combine_matrix_quadrants(
    q11: np.ndarray,
    q12: np.ndarray,
    q21: np.ndarray,
    q22: np.ndarray,
) -> np.ndarray:
    """Assemble partial products into the final result matrix.

    Args:
        q11: Top-left quadrant.
        q12: Top-right quadrant.
        q21: Bottom-left quadrant.
        q22: Bottom-right quadrant.

    Returns:
        The combined full matrix.
    """
    top = np.hstack([q11, q12])
    bottom = np.hstack([q21, q22])
    return np.vstack([top, bottom])
