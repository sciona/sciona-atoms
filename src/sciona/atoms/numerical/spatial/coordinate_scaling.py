"""Explicit per-axis unit conversion and origin shift for coordinate rows."""
import numpy as np
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def _parameters(dimension, numerator, denominator, offset):
    values = tuple(np.asarray(v, dtype=np.float64) for v in (numerator, denominator, offset))
    if dimension < 1 or any(v.shape != (dimension,) or not np.all(np.isfinite(v)) for v in values):
        raise ValueError('Finite per-axis numerator, denominator and offset vectors required')
    if np.any(values[1] == 0):
        raise ValueError('Nonzero denominators required')
    return values


def witness_scale_coordinate_rows(rows: AbstractArray, numerator: object,
                                   denominator: object, offset: object) -> AbstractArray:
    if len(rows.shape) != 2 or rows.shape[0] < 0 or np.dtype(rows.dtype).kind not in 'fiu':
        raise ValueError('Known real coordinate matrix required')
    _parameters(rows.shape[1], numerator, denominator, offset)
    return AbstractArray(shape=rows.shape, dtype='float64')


@register_atom(witness_scale_coordinate_rows)
def scale_coordinate_rows(rows: object, numerator: object,
                          denominator: object, offset: object) -> object:
    """Return float64 coordinate rows multiplied, divided, then shifted per axis.

    Computes (rows*numerator)/denominator-offset in that order, without
    precomputing a scale ratio. Signed scales are allowed. All vectors must
    match the row width; empty row sets are valid. Input is preserved and output
    is newly allocated. No axis reordering, absolute value, geometric clipping,
    orientation inference, unit assumptions, or annotation-sentinel policy.
    """
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.dtype.kind not in 'fiu' or not np.all(np.isfinite(rows)):
        raise ValueError('Finite real coordinate matrix required')
    numerator, denominator, offset = _parameters(rows.shape[1], numerator, denominator, offset)
    with np.errstate(over='ignore', invalid='ignore'):
        result = rows.astype(np.float64) * numerator / denominator - offset
    if not np.all(np.isfinite(result)):
        raise ValueError('Coordinate conversion overflow')
    return result
