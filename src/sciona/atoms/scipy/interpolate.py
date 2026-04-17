"""SciPy interpolate atom wrappers for the general-provider scratch repo."""

from __future__ import annotations

import icontract
import numpy as np
from scipy.interpolate import CubicSpline, RBFInterpolator

from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.registry import register_atom


def _is_array(value: object, *, ndim: int | None = None) -> bool:
    """Return whether ``value`` is a NumPy array with an optional rank constraint."""
    if not isinstance(value, np.ndarray):
        return False
    if ndim is not None and value.ndim != ndim:
        return False
    return np.all(np.isfinite(value))


def _is_axis_index(value: object) -> bool:
    """Return whether ``value`` is an integer axis index."""
    return isinstance(value, (int, np.integer))


def _is_boundary_condition(value: object) -> bool:
    """Return whether ``value`` is a valid boundary-condition specification."""
    return value is None or isinstance(value, (str, tuple))


def _is_extrapolation_mode(value: object) -> bool:
    """Return whether ``value`` is an accepted extrapolation mode."""
    return value is None or isinstance(value, (bool, str))


def _is_optional_int(value: object) -> bool:
    """Return whether ``value`` is ``None`` or an integer."""
    return value is None or isinstance(value, (int, np.integer))


def _is_optional_smoothing(value: object) -> bool:
    """Return whether ``value`` is a valid smoothing parameter."""
    return value is None or isinstance(value, (float, int, np.number, np.ndarray))


def _is_optional_str(value: object) -> bool:
    """Return whether ``value`` is ``None`` or a string."""
    return value is None or isinstance(value, str)


def _is_optional_float(value: object) -> bool:
    """Return whether ``value`` is ``None`` or a numeric scalar."""
    return value is None or isinstance(value, (float, int, np.number))


def witness_cubic_spline_fit(
    x: AbstractArray,
    y: AbstractArray,
    axis: AbstractScalar,
    bc_type: AbstractScalar,
    extrapolate: AbstractScalar,
) -> AbstractArray:
    """Return witness metadata for cubic spline fitting without executing SciPy."""
    _ = (x, y, axis, bc_type, extrapolate)
    return AbstractArray(shape=("callable",), dtype="object")


def witness_rbf_interpolator_fit(
    y: AbstractArray,
    d: AbstractArray,
    neighbors: AbstractScalar,
    smoothing: AbstractArray,
    kernel: AbstractScalar,
    epsilon: AbstractScalar,
    degree: AbstractScalar,
) -> AbstractArray:
    """Return witness metadata for RBF interpolation fitting without executing SciPy."""
    _ = (y, d, neighbors, smoothing, kernel, epsilon, degree)
    return AbstractArray(shape=("callable",), dtype="object")


@register_atom(witness_cubic_spline_fit)  # type: ignore[untyped-decorator]
@icontract.require(lambda x: _is_array(x, ndim=1), "x must be a finite 1D ndarray")
@icontract.require(lambda y: _is_array(y), "y must be a finite ndarray")
@icontract.require(lambda axis: _is_axis_index(axis), "axis must be an integer axis index")
@icontract.require(
    lambda bc_type: _is_boundary_condition(bc_type),
    "bc_type must be a string or tuple boundary specification",
)
@icontract.require(
    lambda extrapolate: _is_extrapolation_mode(extrapolate),
    "extrapolate must be None, bool, or string",
)
@icontract.ensure(lambda result: result is not None, "CubicSplineFit output must not be None")
def cubic_spline_fit(
    x: np.ndarray,
    y: np.ndarray,
    axis: int = 0,
    bc_type: str | tuple = "not-a-knot",
    extrapolate: bool | str | None = None,
) -> CubicSpline:
    """Construct a piecewise cubic spline interpolator."""
    return CubicSpline(x, y, axis=axis, bc_type=bc_type, extrapolate=extrapolate)


@register_atom(witness_rbf_interpolator_fit)  # type: ignore[untyped-decorator]
@icontract.require(lambda y: _is_array(y, ndim=2), "y must be a finite 2D ndarray of coordinates")
@icontract.require(lambda d: _is_array(d), "d must be a finite ndarray of values")
@icontract.require(lambda neighbors: _is_optional_int(neighbors), "neighbors must be None or an integer")
@icontract.require(
    lambda smoothing: _is_optional_smoothing(smoothing),
    "smoothing must be a numeric scalar or an ndarray",
)
@icontract.require(lambda kernel: _is_optional_str(kernel), "kernel must be a string")
@icontract.require(lambda epsilon: _is_optional_float(epsilon), "epsilon must be None or a numeric scalar")
@icontract.require(lambda degree: _is_optional_int(degree), "degree must be None or an integer")
@icontract.ensure(lambda result: result is not None, "RBFInterpolatorFit output must not be None")
def rbf_interpolator_fit(
    y: np.ndarray,
    d: np.ndarray,
    neighbors: int | None = None,
    smoothing: float | np.ndarray = 0.0,
    kernel: str = "thin_plate_spline",
    epsilon: float | None = None,
    degree: int | None = None,
) -> RBFInterpolator:
    """Construct a radial basis function interpolator."""
    return RBFInterpolator(
        y,
        d,
        neighbors=neighbors,
        smoothing=smoothing,
        kernel=kernel,
        epsilon=epsilon,
        degree=degree,
    )


__all__ = [
    "cubic_spline_fit",
    "rbf_interpolator_fit",
    "witness_cubic_spline_fit",
    "witness_rbf_interpolator_fit",
]
