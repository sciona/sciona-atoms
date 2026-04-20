from __future__ import annotations
from typing import Any, Sequence
from sciona.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal


def _as_array_or_scalar(
    shape: tuple[int, ...],
    *,
    dtype: str = "float64",
    min_val: float | None = None,
    max_val: float | None = None,
) -> AbstractArray | AbstractScalar:
    if shape == ():
        return AbstractScalar(dtype=dtype, min_val=min_val, max_val=max_val)
    return AbstractArray(shape=shape, dtype=dtype, min_val=min_val, max_val=max_val)


def _as_array_meta(x: AbstractArray | AbstractScalar) -> AbstractArray:
    if isinstance(x, AbstractArray):
        return x
    return AbstractArray(shape=(), dtype=x.dtype, min_val=x.min_val, max_val=x.max_val)


def _shape_without_axis(shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    if not shape:
        return ()
    ndim = len(shape)
    ax = axis if axis >= 0 else ndim + axis
    if ax < 0 or ax >= ndim:
        raise ValueError(f"axis {axis} out of bounds for shape {shape}")
    return shape[:ax] + shape[ax + 1 :]


def _leading_len(x: AbstractArray) -> int:
    return x.shape[0] if x.shape else 1


def _bounds_len(bounds: AbstractArray | Sequence[Any]) -> int:
    if isinstance(bounds, AbstractArray):
        return bounds.shape[0] if bounds.shape else 1
    return len(bounds)


def witness_scipy_quad(
    func: Any,
    a: float,
    b: float,
    args: tuple = (),
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
) -> tuple[AbstractScalar, AbstractScalar] | tuple[AbstractScalar, AbstractScalar, AbstractScalar]:
    """Describe the integral estimate and error metadata from `scipy.integrate.quad`."""
    _ = (func, a, b, args, epsabs, epsrel, limit, points, weight, wvar, wopts, maxp1, limlst)
    integral = AbstractScalar(dtype="float64")
    abs_err = AbstractScalar(dtype="float64", min_val=0.0)
    if full_output:
        evals = AbstractScalar(dtype="int64", min_val=0.0)
        return (integral, abs_err, evals)
    return (integral, abs_err)


def witness_scipy_solve_ivp(
    fun: Any,
    t_span: tuple[float, float],
    y0: AbstractArray,
    method: str = "RK45",
    t_eval: AbstractArray | None = None,
    dense_output: bool = False,
    events: Any = None,
    vectorized: bool = False,
    args: tuple | None = None,
    **options: Any,
) -> AbstractArray:
    """Describe the state trajectory returned by `scipy.integrate.solve_ivp`."""
    _ = (fun, t_span, method, dense_output, events, vectorized, args, options)
    n_state = _leading_len(y0)
    if t_eval is None:
        return AbstractArray(shape=(n_state,), dtype=y0.dtype)
    n_t = _leading_len(t_eval)
    return AbstractArray(shape=(n_state, n_t), dtype=y0.dtype)


def witness_scipy_simpson(
    y: AbstractArray,
    x: AbstractArray | None = None,
    dx: float = 1.0,
    axis: int = -1,
) -> AbstractArray | AbstractScalar:
    """Describe the reduced integral shape from Simpson-rule integration."""
    _ = (x, dx)
    if len(y.shape) <= 1:
        return AbstractScalar(dtype="float64")
    out_shape = _shape_without_axis(y.shape, axis)
    return _as_array_or_scalar(out_shape, dtype="float64")


def witness_scipy_linalg_solve(
    a: AbstractArray,
    b: AbstractArray,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: str = "gen",
) -> AbstractArray:
    """Describe the solution array returned by `scipy.linalg.solve`."""
    _ = (lower, overwrite_a, overwrite_b, check_finite, assume_a)
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"a must be square 2D, got {a.shape}")
    if not b.shape or b.shape[0] != a.shape[0]:
        raise ValueError(f"Incompatible shapes for solve: a={a.shape}, b={b.shape}")
    return AbstractArray(shape=b.shape, dtype=a.dtype)


def witness_scipy_linalg_inv(
    a: AbstractArray,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> AbstractArray:
    """Describe the inverse of a square matrix."""
    _ = (overwrite_a, check_finite)
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"a must be square 2D, got {a.shape}")
    return AbstractArray(shape=a.shape, dtype=a.dtype)


def witness_scipy_linalg_det(
    a: AbstractArray,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> AbstractArray | AbstractScalar:
    """Describe determinant output for one matrix or a batch of matrices."""
    _ = (overwrite_a, check_finite)
    if len(a.shape) < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"a must be at least 2D with square trailing dims, got {a.shape}")
    out_shape = a.shape[:-2]
    return _as_array_or_scalar(out_shape, dtype="float64")


def witness_scipy_lu_factor(
    a: AbstractArray,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe LU factors and pivot metadata for a square matrix."""
    _ = (overwrite_a, check_finite)
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"a must be square 2D, got {a.shape}")
    n = a.shape[0]
    return (
        AbstractArray(shape=a.shape, dtype=a.dtype),
        AbstractArray(shape=(n,), dtype="int64", min_val=0.0, max_val=float(max(n - 1, 0))),
    )


def witness_scipy_lu_solve(
    lu_and_piv: tuple[AbstractArray, AbstractArray],
    b: AbstractArray,
    trans: int = 0,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> AbstractArray:
    """Describe the solution array produced by `scipy.linalg.lu_solve`."""
    _ = (trans, overwrite_b, check_finite)
    lu, piv = lu_and_piv
    if len(lu.shape) != 2 or lu.shape[0] != lu.shape[1]:
        raise ValueError(f"lu must be square 2D, got {lu.shape}")
    if piv.shape != (lu.shape[0],):
        raise ValueError(f"piv shape must be {(lu.shape[0],)}, got {piv.shape}")
    if not b.shape or b.shape[0] != lu.shape[0]:
        raise ValueError(f"Incompatible shapes for lu_solve: lu={lu.shape}, b={b.shape}")
    return AbstractArray(shape=b.shape, dtype=lu.dtype)


def witness_scipy_minimize(
    fun: Any,
    x0: AbstractArray,
    args: tuple = (),
    method: str | None = None,
    jac: Any = None,
    hess: Any = None,
    hessp: Any = None,
    bounds: Sequence | None = None,
    constraints: Any = (),
    tol: float | None = None,
    callback: Any = None,
    options: dict | None = None,
) -> AbstractArray:
    """Describe the optimized parameter vector returned by `scipy.optimize.minimize`."""
    _ = (fun, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)
    return AbstractArray(shape=x0.shape, dtype="float64")


def witness_scipy_root(
    fun: Any,
    x0: AbstractArray,
    args: tuple = (),
    method: str = "hybr",
    jac: Any = None,
    tol: float | None = None,
    callback: Any = None,
    options: dict | None = None,
) -> AbstractArray:
    """Describe the root estimate returned by `scipy.optimize.root`."""
    _ = (fun, args, method, jac, tol, callback, options)
    return AbstractArray(shape=x0.shape, dtype="float64")


def witness_scipy_linprog(
    c: AbstractArray,
    A_ub: AbstractArray | None = None,
    b_ub: AbstractArray | None = None,
    A_eq: AbstractArray | None = None,
    b_eq: AbstractArray | None = None,
    bounds: Sequence | None = None,
    method: str = "highs",
    callback: Any = None,
    options: dict | None = None,
    x0: AbstractArray | None = None,
) -> AbstractArray:
    """Describe the decision vector returned by linear programming."""
    _ = (A_ub, b_ub, A_eq, b_eq, bounds, method, callback, options, x0)
    n_vars = _leading_len(c)
    return AbstractArray(shape=(n_vars,), dtype="float64")


def witness_scipy_curve_fit(
    f: Any,
    xdata: AbstractArray,
    ydata: AbstractArray,
    p0: AbstractArray | None = None,
    sigma: AbstractArray | None = None,
    absolute_sigma: bool = False,
    check_finite: bool | None = None,
    bounds: Sequence | None = (-float("inf"), float("inf")),
    method: str | None = None,
    jac: Any = None,
    **kwargs: Any,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe fitted parameters and covariance from nonlinear curve fitting."""
    _ = (f, sigma, absolute_sigma, check_finite, bounds, method, jac, kwargs)
    if _leading_len(xdata) != _leading_len(ydata):
        raise ValueError(f"xdata and ydata must have same length, got {xdata.shape} and {ydata.shape}")
    if p0 is None:
        n_params = 1
    else:
        n_params = _leading_len(p0)
    return (
        AbstractArray(shape=(n_params,), dtype="float64"),
        AbstractArray(shape=(n_params, n_params), dtype="float64"),
    )


def witness_scipy_describe(
    a: AbstractArray,
    axis: int | None = 0,
    ddof: int = 1,
    bias: bool = True,
    nan_policy: str = "propagate",
) -> tuple[AbstractScalar, tuple[AbstractScalar, AbstractScalar], AbstractScalar, AbstractScalar, AbstractScalar, AbstractScalar]:
    """Describe summary statistics returned by `scipy.stats.describe`."""
    _ = (a, axis, ddof, bias, nan_policy)
    return (
        AbstractScalar(dtype="int64", min_val=1.0),
        (AbstractScalar(dtype="float64"), AbstractScalar(dtype="float64")),
        AbstractScalar(dtype="float64"),
        AbstractScalar(dtype="float64", min_val=0.0),
        AbstractScalar(dtype="float64"),
        AbstractScalar(dtype="float64"),
    )


def witness_scipy_ttest_ind(
    a: AbstractArray,
    b: AbstractArray,
    axis: int = 0,
    equal_var: bool = True,
    nan_policy: str = "propagate",
    permutations: float | None = None,
    random_state: int | None = None,
    alternative: str = "two-sided",
    trim: float = 0,
) -> tuple[AbstractScalar, AbstractScalar]:
    """Describe the test statistic and p-value from an independent t-test."""
    _ = (axis, equal_var, nan_policy, permutations, random_state, alternative, trim)
    if a.shape != b.shape:
        raise ValueError(f"a and b must have matching shapes, got {a.shape} and {b.shape}")
    return (
        AbstractScalar(dtype="float64"),
        AbstractScalar(dtype="float64", min_val=0.0, max_val=1.0),
    )


def witness_scipy_pearsonr(
    x: AbstractArray,
    y: AbstractArray,
) -> tuple[AbstractScalar, AbstractScalar]:
    """Describe Pearson correlation output."""
    if _leading_len(x) != _leading_len(y):
        raise ValueError(f"x and y must have same length, got {x.shape} and {y.shape}")
    return (
        AbstractScalar(dtype="float64", min_val=-1.0, max_val=1.0),
        AbstractScalar(dtype="float64", min_val=0.0, max_val=1.0),
    )


def witness_scipy_spearmanr(
    a: AbstractArray,
    b: AbstractArray | None = None,
    axis: int | None = 0,
    nan_policy: str = "propagate",
    alternative: str = "two-sided",
) -> tuple[AbstractScalar, AbstractScalar]:
    """Describe Spearman correlation output."""
    _ = (axis, nan_policy, alternative)
    if b is not None and _leading_len(a) != _leading_len(b):
        raise ValueError(f"a and b must have same length, got {a.shape} and {b.shape}")
    return (
        AbstractScalar(dtype="float64", min_val=-1.0, max_val=1.0),
        AbstractScalar(dtype="float64", min_val=0.0, max_val=1.0),
    )


def witness_scipy_norm(
    loc: float = 0.0,
    scale: float = 1.0,
) -> AbstractDistribution:
    """Describe a univariate normal distribution object."""
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    return AbstractDistribution(
        family="normal",
        event_shape=(),
        support_lower=None,
        support_upper=None,
        is_discrete=False,
    )


# ---------------------------------------------------------------------------
# Optimize v2 witnesses (merged from optimize_v2 sub-package)
# ---------------------------------------------------------------------------

def witness_shgoglobaloptimization(
    func: Any,
    bounds: AbstractArray | Sequence[Any],
    args: tuple = (),
    constraints: Any = None,
    n: int = 100,
    iters: int = 1,
    callback: Any = None,
    minimizer_kwargs: dict | None = None,
    options: dict | None = None,
    sampling_method: str | Any = "simplicial",
    *,
    workers: int | Any = 1,
) -> AbstractArray:
    """Describe the SHGO minimizer vector."""
    _ = (func, args, constraints, n, iters, callback, minimizer_kwargs, options, sampling_method, workers)
    return AbstractArray(shape=(_bounds_len(bounds),), dtype="float64")


def witness_differentialevolutionoptimization(
    func: Any,
    bounds: AbstractArray | Sequence[Any],
    args: tuple = (),
    strategy: str | Any = "best1bin",
    maxiter: int = 1000,
    popsize: int = 15,
    tol: float = 0.01,
    mutation: float | tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
    rng: Any = None,
    callback: Any = None,
    disp: bool = False,
    polish: bool = True,
    init: str | AbstractArray = "latinhypercube",
    atol: float = 0.0,
    updating: str = "immediate",
    workers: int | Any = 1,
    constraints: Any = (),
    x0: AbstractArray | None = None,
    *,
    integrality: AbstractArray | None = None,
    vectorized: bool = False,
    seed: Any = None,
) -> AbstractArray:
    """Describe the differential-evolution minimizer vector."""
    _ = (
        func,
        args,
        strategy,
        maxiter,
        popsize,
        tol,
        mutation,
        recombination,
        rng,
        callback,
        disp,
        polish,
        init,
        atol,
        updating,
        workers,
        constraints,
        x0,
        integrality,
        vectorized,
        seed,
    )
    return AbstractArray(shape=(_bounds_len(bounds),), dtype="float64")


# ---------------------------------------------------------------------------
# Sparse graph v2 witnesses (merged from sparse_graph_v2 sub-package)
# ---------------------------------------------------------------------------

def witness_singlesourceshortestpath(
    csgraph: AbstractArray,
    indices: AbstractArray | AbstractScalar = AbstractScalar(dtype="int64", min_val=0.0),
    method: str = "auto",
    directed: bool = True,
    return_predecessors: bool = False,
    unweighted: bool = False,
    overwrite: bool = False,
) -> AbstractArray | tuple[AbstractArray, AbstractArray]:
    """Return metadata for source-indexed SciPy shortest-path output."""
    _ = (method, directed, unweighted, overwrite)
    if isinstance(indices, AbstractArray) and indices.shape:
        output_shape = (indices.shape[0], csgraph.shape[0])
    else:
        output_shape = (csgraph.shape[0],)
    distances = AbstractArray(shape=output_shape, dtype="float64")
    if return_predecessors:
        predecessors = AbstractArray(shape=output_shape, dtype="int64", min_val=-9999.0)
        return (distances, predecessors)
    return distances


def witness_allpairsshortestpath(
    csgraph: AbstractArray,
    method: str = "auto",
    directed: bool = True,
    return_predecessors: bool = False,
    unweighted: bool = False,
    overwrite: bool = False,
) -> AbstractArray | tuple[AbstractArray, AbstractArray]:
    """Return metadata for all-pairs SciPy shortest-path output."""
    _ = (method, directed, unweighted, overwrite)
    distances = AbstractArray(shape=csgraph.shape, dtype="float64")
    if return_predecessors:
        predecessors = AbstractArray(shape=csgraph.shape, dtype="int64", min_val=-9999.0)
        return (distances, predecessors)
    return distances


def witness_minimumspanningtree(
    csgraph: AbstractArray,
    overwrite: bool = False,
) -> AbstractArray:
    """Return sparse-matrix-shaped metadata for the MST wrapper."""
    _ = overwrite
    return AbstractArray(shape=csgraph.shape, dtype="float64")
