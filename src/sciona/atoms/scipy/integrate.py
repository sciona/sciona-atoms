"""SciPy integrate atom wrappers for the general-provider scratch repo."""

from __future__ import annotations

from typing import Callable, Sequence, TypeAlias

import icontract
import numpy as np
import scipy.integrate
from scipy.integrate._ivp.ivp import OdeResult

from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.registry import register_atom

ArrayLike: TypeAlias = np.ndarray | list[object] | tuple[object, ...]
QuadInfo: TypeAlias = dict[str, object]
QuadResult: TypeAlias = (
    tuple[float, float]
    | tuple[float, float, QuadInfo]
    | tuple[float, float, QuadInfo, str]
)
SolveIVPEvent: TypeAlias = Callable[..., np.ndarray | float]
SolveIVPOptionsValue: TypeAlias = (
    float
    | int
    | bool
    | str
    | np.ndarray
    | Sequence[float]
    | tuple[float, float]
    | tuple[int, int]
    | tuple[object, ...]
    | SolveIVPEvent
    | None
)


def witness_scipy_quad(
    func: object,
    a: float,
    b: float,
    args: tuple[object, ...] = (),
    full_output: int = 0,
    epsabs: float = 1.49e-8,
    epsrel: float = 1.49e-8,
    limit: int = 50,
    points: Sequence | None = None,
    weight: str | None = None,
    wvar: float | complex | None = None,
    wopts: tuple | None = None,
    maxp1: int = 50,
    limlst: int = 50,
    complex_func: bool = False,
) -> tuple[AbstractScalar, AbstractScalar] | tuple[AbstractScalar, AbstractScalar, AbstractScalar]:
    """Return witness metadata for quadrature without executing SciPy."""
    _ = (func, a, b, args, epsabs, epsrel, limit, points, weight, wvar, wopts, maxp1, limlst, complex_func)
    integral = AbstractScalar(dtype="float64")
    abs_err = AbstractScalar(dtype="float64", min_val=0.0)
    if full_output:
        evals = AbstractScalar(dtype="int64", min_val=0.0)
        return (integral, abs_err, evals)
    return (integral, abs_err)


def witness_scipy_solve_ivp(
    fun: object,
    t_span: tuple[float, float],
    y0: AbstractArray,
    method: str = "RK45",
    t_eval: AbstractArray | None = None,
    dense_output: bool = False,
    events: object = None,
    vectorized: bool = False,
    args: tuple | None = None,
    **options: object,
) -> AbstractArray:
    """Return witness metadata for IVP solving without executing SciPy."""
    _ = (fun, t_span, method, dense_output, events, vectorized, args, options)
    n_state = y0.shape[0] if y0.shape else 1
    if t_eval is None:
        return AbstractArray(shape=(n_state,), dtype=y0.dtype)
    n_t = t_eval.shape[0] if t_eval.shape else 1
    return AbstractArray(shape=(n_state, n_t), dtype=y0.dtype)


def witness_scipy_simpson(
    y: AbstractArray,
    x: AbstractArray | None = None,
    dx: float = 1.0,
    axis: int = -1,
) -> AbstractArray | AbstractScalar:
    """Return witness metadata for Simpson integration without executing SciPy."""
    _ = (x, dx)
    if len(y.shape) <= 1:
        return AbstractScalar(dtype="float64")
    ax = axis if axis >= 0 else len(y.shape) + axis
    out_shape = y.shape[:ax] + y.shape[ax + 1 :]
    if not out_shape:
        return AbstractScalar(dtype="float64")
    return AbstractArray(shape=out_shape, dtype="float64")


@register_atom(witness_scipy_quad)  # type: ignore[untyped-decorator]
@icontract.require(lambda func: func is not None, "Function must not be None")
@icontract.ensure(lambda result: len(result) >= 2, "Result must contain at least (y, abserr)")
def quad(
    func: Callable[..., float],
    a: float,
    b: float,
    args: tuple[object, ...] = (),
    full_output: int = 0,
    epsabs: float = 1.49e-8,
    epsrel: float = 1.49e-8,
    limit: int = 50,
    points: Sequence | None = None,
    weight: str | None = None,
    wvar: float | complex | None = None,
    wopts: tuple | None = None,
    maxp1: int = 50,
    limlst: int = 50,
    complex_func: bool = False,
) -> QuadResult:
    """Compute a definite integral."""
    return scipy.integrate.quad(
        func,
        a,
        b,
        args=args,
        full_output=full_output,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
        points=points,
        weight=weight,
        wvar=wvar,
        wopts=wopts,
        maxp1=maxp1,
        limlst=limlst,
        complex_func=complex_func,
    )


@register_atom(witness_scipy_solve_ivp)  # type: ignore[untyped-decorator]
@icontract.require(
    lambda fun, t_span, y0: fun is not None and t_span is not None and y0 is not None,
    "ODE function, time span, and initial condition must not be None",
)
@icontract.ensure(lambda result: result is not None, "ODE solution result must not be None")
def solve_ivp(
    fun: SolveIVPEvent,
    t_span: tuple[float, float],
    y0: ArrayLike,
    method: str = "RK45",
    t_eval: ArrayLike | None = None,
    dense_output: bool = False,
    events: SolveIVPEvent | Sequence[SolveIVPEvent] | None = None,
    vectorized: bool = False,
    args: tuple[object, ...] | None = None,
    **options: SolveIVPOptionsValue,
) -> OdeResult:
    """Solve an initial value problem for an ODE system."""
    return scipy.integrate.solve_ivp(
        fun,
        t_span,
        y0,
        method=method,
        t_eval=t_eval,
        dense_output=dense_output,
        events=events,
        vectorized=vectorized,
        args=args,
        **options,
    )


@register_atom(witness_scipy_simpson)  # type: ignore[untyped-decorator]
@icontract.require(lambda y: np.asarray(y).ndim >= 1, "Input y must be at least 1D")
@icontract.require(lambda y: len(np.asarray(y)) > 0, "Input y must not be empty")
@icontract.ensure(lambda result: result is not None, "Integration result must not be None")
def simpson(
    y: ArrayLike,
    x: ArrayLike | None = None,
    dx: float = 1.0,
    axis: int = -1,
) -> float | np.ndarray:
    """Integrate samples using Simpson's rule."""
    return scipy.integrate.simpson(y, x=x, dx=dx, axis=axis)


__all__ = ["quad", "solve_ivp", "simpson", "witness_scipy_quad", "witness_scipy_solve_ivp", "witness_scipy_simpson"]
