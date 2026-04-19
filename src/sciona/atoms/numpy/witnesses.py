from __future__ import annotations
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal, AbstractRNGState
from typing import Any, Sequence
ShapeLike = int | Sequence[int] | None
AbstractValue = AbstractArray | AbstractScalar


def _dtype_name(dtype: Any, default: str = "float64") -> str:
    if dtype is None:
        return default
    return str(dtype)


def _normalize_shape(shape: ShapeLike) -> tuple[int, ...]:
    if shape is None:
        return ()
    if isinstance(shape, int):
        return (shape,)
    return tuple(int(dim) for dim in shape)


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _as_array_meta(x: AbstractValue) -> AbstractArray:
    if isinstance(x, AbstractArray):
        return x
    return AbstractArray(
        shape=(),
        dtype=x.dtype,
        min_val=x.min_val,
        max_val=x.max_val,
    )


def _as_array_or_scalar(
    shape: tuple[int, ...],
    *,
    dtype: str,
    min_val: float | None = None,
    max_val: float | None = None,
    is_sorted: bool = False,
) -> AbstractArray | AbstractScalar:
    if shape == ():
        return AbstractScalar(dtype=dtype, min_val=min_val, max_val=max_val)
    return AbstractArray(
        shape=shape,
        dtype=dtype,
        min_val=min_val,
        max_val=max_val,
        is_sorted=is_sorted,
    )


def _dtype_is_complex(dtype: str) -> bool:
    return "complex" in dtype or dtype in {"csingle", "cdouble", "clongdouble"}


def _dtype_is_integer_like(dtype: str) -> bool:
    return dtype.startswith("int") or dtype.startswith("uint") or dtype in {"bool", "bool_"}


def _emath_real_dtype(dtype: str) -> str:
    if _dtype_is_complex(dtype):
        return dtype
    if dtype.startswith("float"):
        return dtype
    return "float64"


def _emath_complex_dtype(dtype: str) -> str:
    if dtype in {"float16", "float32", "int8", "int16", "uint8", "uint16", "complex64", "csingle"}:
        return "complex64"
    return "complex128"


def _may_contain_negative_real(arr: AbstractArray) -> bool:
    if _dtype_is_complex(arr.dtype):
        return False
    return arr.min_val is None or arr.min_val < 0.0


def witness_np_array(
    object: AbstractValue,  # noqa: A002
    dtype: Any = None,
    *,
    copy: bool | None = True,
    order: str = "K",
    subok: bool = False,
    ndmin: int = 0,
    ndmax: int = 0,
    like: AbstractValue | None = None,
) -> AbstractArray:
    """Describe `numpy.array` output metadata after optional rank promotion."""
    arr = _as_array_meta(object)
    out_shape = arr.shape
    while len(out_shape) < ndmin:
        out_shape = (1,) + out_shape
    return AbstractArray(
        shape=out_shape,
        dtype=_dtype_name(dtype, arr.dtype),
        min_val=arr.min_val,
        max_val=arr.max_val,
        is_sorted=arr.is_sorted,
    )


def witness_np_zeros(
    shape: ShapeLike,
    dtype: Any = None,
    order: str = "C",
    *,
    device: str | None = None,
    like: AbstractValue | None = None,
) -> AbstractArray:
    """Describe a zero-filled array with the requested shape and dtype."""
    return AbstractArray(
        shape=_normalize_shape(shape),
        dtype=_dtype_name(dtype),
        min_val=0.0,
        max_val=0.0,
        is_sorted=True,
    )


def witness_np_dot(
    a: AbstractValue,
    b: AbstractValue,
    out: AbstractArray | None = None,
) -> AbstractArray | AbstractScalar:
    """Describe the output shape for NumPy dot products across common ranks."""
    a_arr = _as_array_meta(a)
    b_arr = _as_array_meta(b)
    a_shape = a_arr.shape
    b_shape = b_arr.shape

    if a_shape == () and b_shape == ():
        return AbstractScalar(dtype=a_arr.dtype)
    if a_shape == ():
        return AbstractArray(shape=b_shape, dtype=b_arr.dtype, min_val=b_arr.min_val, max_val=b_arr.max_val)
    if b_shape == ():
        return AbstractArray(shape=a_shape, dtype=a_arr.dtype, min_val=a_arr.min_val, max_val=a_arr.max_val)

    if len(a_shape) == 1 and len(b_shape) == 1:
        if a_shape[0] != b_shape[0]:
            raise ValueError(f"Incompatible dot dimensions: {a_shape} and {b_shape}")
        return AbstractScalar(dtype=a_arr.dtype)
    if len(a_shape) == 2 and len(b_shape) == 2:
        if a_shape[1] != b_shape[0]:
            raise ValueError(f"Incompatible dot dimensions: {a_shape} and {b_shape}")
        return AbstractArray(shape=(a_shape[0], b_shape[1]), dtype=a_arr.dtype)
    if len(a_shape) == 2 and len(b_shape) == 1:
        if a_shape[1] != b_shape[0]:
            raise ValueError(f"Incompatible dot dimensions: {a_shape} and {b_shape}")
        return AbstractArray(shape=(a_shape[0],), dtype=a_arr.dtype)
    if len(a_shape) == 1 and len(b_shape) == 2:
        if a_shape[0] != b_shape[0]:
            raise ValueError(f"Incompatible dot dimensions: {a_shape} and {b_shape}")
        return AbstractArray(shape=(b_shape[1],), dtype=a_arr.dtype)

    raise ValueError(f"Unsupported dot dimensions for witness: {a_shape} and {b_shape}")


def witness_np_vstack(
    tup: Sequence[AbstractValue],
    dtype: Any = None,
    casting: str = "same_kind",
) -> AbstractArray:
    """Describe the stacked array produced by vertical concatenation."""
    if not tup:
        raise ValueError("tup must be non-empty")

    shapes: list[tuple[int, ...]] = []
    for val in tup:
        shape = _as_array_meta(val).shape
        if shape == ():
            shapes.append((1, 1))
        elif len(shape) == 1:
            shapes.append((1, shape[0]))
        else:
            shapes.append(shape)

    tail = shapes[0][1:]
    for shape in shapes[1:]:
        if shape[1:] != tail:
            raise ValueError(f"vstack shape mismatch: {shapes[0]} vs {shape}")

    out_rows = sum(shape[0] for shape in shapes)
    return AbstractArray(
        shape=(out_rows,) + tail,
        dtype=_dtype_name(dtype, _as_array_meta(tup[0]).dtype),
    )


def witness_np_reshape(
    a: AbstractArray,
    shape: ShapeLike,
    order: str = "C",
    *,
    copy: bool | None = None,
) -> AbstractArray:
    """Describe the reshaped array while preserving dtype and value bounds."""
    src_shape = a.shape
    src_total = _numel(src_shape)
    tgt = list(_normalize_shape(shape))

    unknown_count = tgt.count(-1)
    if unknown_count > 1:
        raise ValueError("shape can have at most one -1")

    if unknown_count == 1:
        known_prod = 1
        for dim in tgt:
            if dim != -1:
                known_prod *= dim
        if known_prod == 0:
            inferred = 0
        else:
            if src_total % known_prod != 0:
                raise ValueError("shape is incompatible with input size")
            inferred = src_total // known_prod
        tgt[tgt.index(-1)] = inferred
    else:
        if src_total != _numel(tuple(tgt)):
            raise ValueError("shape is incompatible with input size")

    return AbstractArray(
        shape=tuple(tgt),
        dtype=a.dtype,
        min_val=a.min_val,
        max_val=a.max_val,
        is_sorted=a.is_sorted,
    )


def witness_np_emath_sqrt(x: AbstractValue) -> AbstractArray | AbstractScalar:
    """Describe elementwise square-root output, widening to complex when needed."""
    arr = _as_array_meta(x)
    promotes_complex = _may_contain_negative_real(arr)
    out_dtype = _emath_complex_dtype(arr.dtype) if promotes_complex else _emath_real_dtype(arr.dtype)
    min_val = 0.0 if not promotes_complex and not _dtype_is_complex(arr.dtype) else None
    return _as_array_or_scalar(
        arr.shape,
        dtype=out_dtype,
        min_val=min_val,
        max_val=None,
    )


def witness_np_emath_log(x: AbstractValue) -> AbstractArray | AbstractScalar:
    """Describe elementwise natural-log output, widening to complex when needed."""
    arr = _as_array_meta(x)
    out_dtype = _emath_complex_dtype(arr.dtype) if _may_contain_negative_real(arr) else _emath_real_dtype(arr.dtype)
    return _as_array_or_scalar(arr.shape, dtype=out_dtype)


def witness_np_emath_log10(x: AbstractValue) -> AbstractArray | AbstractScalar:
    """Describe elementwise base-10 logarithm output."""
    return witness_np_emath_log(x)


def witness_np_emath_logn(n: AbstractValue, x: AbstractValue) -> AbstractArray | AbstractScalar:
    """Describe elementwise logarithm output under an arbitrary base."""
    base = _as_array_meta(n)
    arr = _as_array_meta(x)
    promotes_complex = _may_contain_negative_real(arr) or _may_contain_negative_real(base)
    out_dtype = _emath_complex_dtype(arr.dtype) if promotes_complex else _emath_real_dtype(arr.dtype)
    return _as_array_or_scalar(arr.shape, dtype=out_dtype)


def witness_np_emath_power(
    x: AbstractValue,
    p: AbstractValue | Any,
) -> AbstractArray | AbstractScalar:
    """Describe elementwise power output while preserving the input shape."""
    arr = _as_array_meta(x)
    power_arr = _as_array_meta(p) if isinstance(p, (AbstractArray, AbstractScalar)) else None
    if _may_contain_negative_real(arr):
        out_dtype = _emath_complex_dtype(arr.dtype)
    elif power_arr is not None and _dtype_is_integer_like(arr.dtype) and (
        power_arr.min_val is None or power_arr.min_val < 0.0
    ):
        out_dtype = "float64"
    else:
        out_dtype = arr.dtype
    return _as_array_or_scalar(arr.shape, dtype=out_dtype)


def witness_np_linalg_solve(
    a: AbstractArray,
    b: AbstractValue,
) -> AbstractArray:
    """Describe the solution array returned by a square linear solve."""
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"a must be square 2D, got {a.shape}")
    b_arr = _as_array_meta(b)
    if not b_arr.shape:
        raise ValueError("b must have at least one dimension")
    if b_arr.shape[0] != a.shape[0]:
        raise ValueError(f"Dimension mismatch between a={a.shape} and b={b_arr.shape}")
    return AbstractArray(
        shape=b_arr.shape,
        dtype=a.dtype,
        min_val=b_arr.min_val,
        max_val=b_arr.max_val,
    )


def witness_np_linalg_inv(a: AbstractArray) -> AbstractArray:
    """Describe the inverse of a square matrix."""
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"a must be square 2D, got {a.shape}")
    return AbstractArray(shape=a.shape, dtype=a.dtype)


def witness_np_linalg_det(a: AbstractArray) -> AbstractArray | AbstractScalar:
    """Describe determinant output for a matrix or batch of matrices."""
    if len(a.shape) < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"a must be at least 2D with square trailing dims, got {a.shape}")
    out_shape = a.shape[:-2]
    return _as_array_or_scalar(out_shape, dtype="float64")


def witness_np_linalg_norm(
    x: AbstractValue,
    ord: int | float | str | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> AbstractScalar:
    """Describe a non-negative norm scalar."""
    _ = x, ord, axis, keepdims
    return AbstractScalar(dtype="float64", min_val=0.0)


def witness_np_polyval(
    x: AbstractValue,
    c: AbstractArray,
) -> AbstractArray | AbstractScalar:
    """Describe polynomial evaluation output over the sample shape."""
    if not c.shape or c.shape[0] <= 0:
        raise ValueError("c must be non-empty")
    arr = _as_array_meta(x)
    return _as_array_or_scalar(arr.shape, dtype=arr.dtype)


def witness_np_polyfit(
    x: AbstractArray,
    y: AbstractArray,
    deg: int,
) -> AbstractArray:
    """Describe fitted polynomial coefficients for the requested degree."""
    if not x.shape or not y.shape or x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y must have matching leading dims, got {x.shape} and {y.shape}")
    if deg < 0:
        raise ValueError("deg must be non-negative")
    return AbstractArray(shape=(deg + 1,), dtype="float64")


def witness_np_polyder(
    c: AbstractArray,
    m: int = 1,
) -> AbstractArray:
    """Describe polynomial derivative coefficients."""
    n = c.shape[0] if c.shape else 1
    out_n = max(1, n - m)
    return AbstractArray(shape=(out_n,), dtype=c.dtype)


def witness_np_polyint(
    c: AbstractArray,
    m: int = 1,
    k: AbstractValue | float = 0,
) -> AbstractArray:
    """Describe polynomial integral coefficients."""
    _ = k
    n = c.shape[0] if c.shape else 1
    return AbstractArray(shape=(n + m,), dtype=c.dtype)


def witness_np_polyadd(
    c1: AbstractArray,
    c2: AbstractArray,
) -> AbstractArray:
    """Describe coefficient output for polynomial addition."""
    n1 = c1.shape[0] if c1.shape else 1
    n2 = c2.shape[0] if c2.shape else 1
    return AbstractArray(shape=(max(n1, n2),), dtype=c1.dtype)


def witness_np_polymul(
    c1: AbstractArray,
    c2: AbstractArray,
) -> AbstractArray:
    """Describe coefficient output for polynomial multiplication."""
    n1 = c1.shape[0] if c1.shape else 1
    n2 = c2.shape[0] if c2.shape else 1
    return AbstractArray(shape=(n1 + n2 - 1,), dtype=c1.dtype)


def witness_np_polyroots(c: AbstractArray) -> AbstractArray:
    """Describe the complex roots returned for a polynomial."""
    n = c.shape[0] if c.shape else 1
    if n < 2:
        raise ValueError("Polynomial degree must be at least 1 to have roots")
    return AbstractArray(shape=(n - 1,), dtype="complex128")


def witness_np_rand(
    *dims: int | Sequence[int],
    size: ShapeLike = None,
    seed: Any = None,
    rng: AbstractRNGState | None = None,
) -> AbstractArray | AbstractScalar:
    """Describe uniform random samples in the half-open interval [0, 1)."""
    _ = seed, rng
    if dims and size is not None:
        raise ValueError("Provide either NumPy-style dimension arguments or size, not both")
    if size is not None:
        out_shape = _normalize_shape(size)
    elif not dims:
        out_shape = ()
    elif len(dims) == 1 and not isinstance(dims[0], int):
        out_shape = _normalize_shape(dims[0])
    else:
        out_shape = tuple(int(dim) for dim in dims)
    return _as_array_or_scalar(
        out_shape,
        dtype="float64",
        min_val=0.0,
        max_val=1.0,
    )


def witness_np_uniform(
    low: Any = 0.0,
    high: Any = 1.0,
    size: ShapeLike = None,
    seed: Any = None,
    rng: AbstractRNGState | None = None,
) -> AbstractArray | AbstractScalar:
    """Describe uniformly distributed samples over the requested interval."""
    _ = seed, rng
    if size is None:
        try:
            import numpy as _np

            broadcast_shape = _np.broadcast(_np.asarray(low), _np.asarray(high)).shape
            out_shape = () if broadcast_shape == () else tuple(int(dim) for dim in broadcast_shape)
            min_val = float(_np.min(low))
            max_val = float(_np.max(high))
        except Exception:
            out_shape = ()
            min_val = None
            max_val = None
    else:
        out_shape = _normalize_shape(size)
        try:
            import numpy as _np

            min_val = float(_np.min(low))
            max_val = float(_np.max(high))
        except Exception:
            min_val = None
            max_val = None
    return _as_array_or_scalar(
        out_shape,
        dtype="float64",
        min_val=min_val,
        max_val=max_val,
    )


def witness_np_default_rng(seed: Any = None) -> AbstractRNGState:
    """Describe a freshly initialized NumPy random generator state."""
    seed_value = seed if isinstance(seed, int) else 0
    return AbstractRNGState(seed=seed_value, consumed=0, is_split=False)


def witness_np_fftfreq(
    n: int,
    d: float = 1.0,
) -> AbstractArray:
    """Ghost witness for numpy.fft.fftfreq.

    Postconditions:
        - Output shape is (n,).
        - Output dtype is float64.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    return AbstractArray(shape=(n,), dtype="float64")


def witness_np_fftshift(
    x: AbstractArray,
    axes: int | Sequence[int] | None = None,
) -> AbstractArray:
    """Ghost witness for numpy.fft.fftshift.

    Postconditions:
        - Output shape matches input shape.
        - Output dtype matches input dtype.
    """
    _ = axes
    return AbstractArray(shape=x.shape, dtype=x.dtype)
