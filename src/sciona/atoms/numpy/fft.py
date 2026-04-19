"""NumPy FFT atom wrappers for the general provider."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import icontract
from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.registry import register_atom

ArrayLike = np.ndarray | list[object] | tuple[object, ...]


def _shape_from_value(value: AbstractArray | AbstractScalar) -> tuple[int, ...]:
    if isinstance(value, AbstractArray):
        return value.shape
    return ()


def _concrete_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _axis_index(shape: tuple[int, ...], axis: object) -> int | None:
    if not shape:
        return None
    concrete_axis = _concrete_int(axis)
    return (concrete_axis if concrete_axis is not None else -1) % len(shape)


def _shape_with_axis_length(
    shape: tuple[int, ...],
    length: int | None,
    axis: object = -1,
) -> tuple[int, ...]:
    axis_index = _axis_index(shape, axis)
    if axis_index is None or length is None:
        return shape
    parts = list(shape)
    parts[axis_index] = length
    return tuple(parts)


def _fftn_shape(
    shape: tuple[int, ...],
    s: Sequence[int] | AbstractArray | None = None,
    axes: Sequence[int] | AbstractArray | None = None,
) -> tuple[int, ...]:
    if not shape or not isinstance(s, Sequence):
        return shape
    if isinstance(axes, Sequence):
        target_axes = tuple(int(axis) % len(shape) for axis in axes)
    else:
        target_axes = tuple(range(len(shape) - len(s), len(shape)))
    if len(target_axes) != len(s):
        return shape
    out_shape = list(shape)
    for axis, length in zip(target_axes, s, strict=True):
        out_shape[axis] = shape[axis] if int(length) == -1 else int(length)
    return tuple(out_shape)


def witness_fft(
    a: AbstractArray | AbstractScalar,
    n: int | AbstractScalar | None = None,
    axis: int | AbstractScalar | None = None,
    norm: AbstractScalar | None = None,
) -> AbstractArray | AbstractScalar:
    shape = _shape_from_value(a)
    out_shape = _shape_with_axis_length(shape, _concrete_int(n), axis if axis is not None else -1)
    if out_shape == ():
        return AbstractScalar(dtype="complex128")
    return AbstractArray(shape=out_shape, dtype="complex128")


def witness_ifft(
    a: AbstractArray | AbstractScalar,
    n: int | AbstractScalar | None = None,
    axis: int | AbstractScalar | None = None,
    norm: AbstractScalar | None = None,
) -> AbstractArray | AbstractScalar:
    return witness_fft(a, n=n, axis=axis, norm=norm)


def witness_rfft(
    a: AbstractArray | AbstractScalar,
    n: int | AbstractScalar | None = None,
    axis: int | AbstractScalar | None = None,
    norm: AbstractScalar | None = None,
) -> AbstractArray | AbstractScalar:
    shape = _shape_from_value(a)
    if shape == ():
        return AbstractScalar(dtype="complex128")
    axis_index = _axis_index(shape, axis if axis is not None else -1)
    input_length = _concrete_int(n) or shape[axis_index if axis_index is not None else -1]
    out_length = input_length // 2 + 1
    return AbstractArray(
        shape=_shape_with_axis_length(shape, out_length, axis if axis is not None else -1),
        dtype="complex128",
    )


def witness_irfft(
    a: AbstractArray | AbstractScalar,
    n: int | AbstractScalar | None = None,
    axis: int | AbstractScalar | None = None,
    norm: AbstractScalar | None = None,
) -> AbstractArray | AbstractScalar:
    shape = _shape_from_value(a)
    if shape == ():
        return AbstractScalar(dtype="float64")
    axis_index = _axis_index(shape, axis if axis is not None else -1)
    input_length = shape[axis_index if axis_index is not None else -1]
    out_length = _concrete_int(n)
    if out_length is None:
        out_length = max(0, 2 * (input_length - 1))
    return AbstractArray(
        shape=_shape_with_axis_length(shape, out_length, axis if axis is not None else -1),
        dtype="float64",
    )


def witness_fftfreq(n: int | AbstractScalar, d: AbstractScalar | None = None) -> AbstractArray:
    concrete_n = _concrete_int(n)
    return AbstractArray(shape=(concrete_n if concrete_n is not None else 1,), dtype="float64")


def witness_fftn(
    a: AbstractArray | AbstractScalar,
    s: Sequence[int] | AbstractArray | None = None,
    axes: Sequence[int] | AbstractArray | None = None,
    norm: AbstractScalar | None = None,
) -> AbstractArray | AbstractScalar:
    shape = _shape_from_value(a)
    out_shape = _fftn_shape(shape, s=s, axes=axes)
    if out_shape == ():
        return AbstractScalar(dtype="complex128")
    return AbstractArray(shape=out_shape, dtype="complex128")


def witness_ifftn(
    a: AbstractArray | AbstractScalar,
    s: Sequence[int] | AbstractArray | None = None,
    axes: Sequence[int] | AbstractArray | None = None,
    norm: AbstractScalar | None = None,
) -> AbstractArray | AbstractScalar:
    return witness_fftn(a, s=s, axes=axes, norm=norm)


def witness_hfft(
    a: AbstractArray | AbstractScalar,
    n: int | AbstractScalar | None = None,
    axis: int | AbstractScalar | None = None,
    norm: AbstractScalar | None = None,
) -> AbstractArray | AbstractScalar:
    shape = _shape_from_value(a)
    if shape == ():
        return AbstractScalar(dtype="float64")
    axis_index = _axis_index(shape, axis if axis is not None else -1)
    input_length = shape[axis_index if axis_index is not None else -1]
    out_length = _concrete_int(n)
    if out_length is None:
        out_length = max(0, 2 * (input_length - 1))
    return AbstractArray(
        shape=_shape_with_axis_length(shape, out_length, axis if axis is not None else -1),
        dtype="float64",
    )


def witness_fftshift(
    x: AbstractArray | AbstractScalar,
    axes: AbstractScalar | AbstractArray | None = None,
) -> AbstractArray | AbstractScalar:
    if isinstance(x, AbstractScalar):
        return AbstractScalar(dtype=x.dtype)
    return AbstractArray(shape=x.shape, dtype=x.dtype)


@register_atom(witness_fft)  # type: ignore[untyped-decorator]
@icontract.require(lambda a: a is not None, "Input array must not be None")
@icontract.require(lambda a: np.asarray(a).size > 0, "Input array must not be empty")
@icontract.ensure(
    lambda result, a, n, axis: result.shape[axis] == (n if n is not None else np.asarray(a).shape[axis]),
    "Result shape must match n or input shape",
)
@icontract.ensure(lambda result: np.iscomplexobj(result), "FFT output must be complex-valued")
def fft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the one-dimensional discrete Fourier transform."""
    return np.fft.fft(a, n=n, axis=axis, norm=norm, out=out)


@register_atom(witness_ifft)  # type: ignore[untyped-decorator]
@icontract.require(lambda a: a is not None, "Input array must not be None")
@icontract.require(lambda a: np.asarray(a).size > 0, "Input array must not be empty")
@icontract.ensure(
    lambda result, a, n, axis: result.shape[axis] == (n if n is not None else np.asarray(a).shape[axis]),
    "Result shape must match n or input shape",
)
def ifft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the inverse one-dimensional discrete Fourier transform."""
    return np.fft.ifft(a, n=n, axis=axis, norm=norm, out=out)


@register_atom(witness_rfft)  # type: ignore[untyped-decorator]
@icontract.require(lambda a: a is not None, "Input array must not be None")
@icontract.require(lambda a: np.asarray(a).size > 0, "Input array must not be empty")
@icontract.ensure(
    lambda result, a, n, axis: result.shape[axis]
    == ((n if n is not None else np.asarray(a).shape[axis]) // 2 + 1),
    "Result shape must match n//2+1 or input shape",
)
@icontract.ensure(lambda result: np.iscomplexobj(result), "RFFT output must be complex-valued")
def rfft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the FFT for real-valued input."""
    return np.fft.rfft(a, n=n, axis=axis, norm=norm, out=out)


@register_atom(witness_irfft)  # type: ignore[untyped-decorator]
@icontract.require(lambda a: a is not None, "Input array must not be None")
@icontract.require(lambda a: np.asarray(a).size > 0, "Input array must not be empty")
@icontract.ensure(
    lambda result, a, n, axis: result.shape[axis] == (n if n is not None else 2 * (np.asarray(a).shape[axis] - 1)),
    "Result shape must match n or inferred input shape",
)
@icontract.ensure(lambda result: np.isrealobj(result), "IRFFT output must be real-valued")
def irfft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the inverse FFT for a real-input spectrum."""
    return np.fft.irfft(a, n=n, axis=axis, norm=norm, out=out)


@register_atom(witness_fftfreq)  # type: ignore[untyped-decorator]
@icontract.ensure(lambda result, n: result.shape == (n,), "Result shape must match n")
def fftfreq(n: int, d: float = 1.0, device: str | None = None) -> np.ndarray:
    """Return the discrete Fourier transform sample frequencies."""
    return np.fft.fftfreq(n, d=d, device=device)


@register_atom(witness_fftn)  # type: ignore[untyped-decorator]
@icontract.require(lambda a: a is not None, "Input array must not be None")
@icontract.require(lambda a: np.asarray(a).size > 0, "Input array must not be empty")
@icontract.ensure(lambda result: np.iscomplexobj(result), "FFTN output must be complex-valued")
def fftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the N-dimensional discrete Fourier transform."""
    return np.fft.fftn(a, s=s, axes=axes, norm=norm, out=out)


@register_atom(witness_ifftn)  # type: ignore[untyped-decorator]
@icontract.require(lambda a: a is not None, "Input array must not be None")
@icontract.require(lambda a: np.asarray(a).size > 0, "Input array must not be empty")
@icontract.ensure(lambda result: np.iscomplexobj(result), "IFFTN output must be complex-valued")
def ifftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the inverse N-dimensional discrete Fourier transform."""
    return np.fft.ifftn(a, s=s, axes=axes, norm=norm, out=out)


@register_atom(witness_hfft)  # type: ignore[untyped-decorator]
@icontract.require(lambda a: a is not None, "Input array must not be None")
@icontract.ensure(
    lambda result, a, n, axis: result.shape[axis]
    == (n if n is not None else 2 * (np.asarray(a).shape[axis] - 1)),
    "Result shape must match n or inferred Hermitian input shape",
)
@icontract.ensure(lambda result: np.isrealobj(result), "HFFT output must be real-valued")
def hfft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the FFT of a signal with Hermitian symmetry."""
    return np.fft.hfft(a, n=n, axis=axis, norm=norm, out=out)


@register_atom(witness_fftshift)  # type: ignore[untyped-decorator]
@icontract.require(lambda x: x is not None, "Input must not be None")
@icontract.ensure(lambda result, x: result.shape == np.asarray(x).shape, "Result shape must match input shape")
def fftshift(x: ArrayLike, axes: int | tuple[int, ...] | None = None) -> np.ndarray:
    """Shift the zero-frequency component to the center of the spectrum."""
    return np.fft.fftshift(x, axes=axes)


forwardmultidimensionalfft = fftn
inversemultidimensionalfft = ifftn
hermitianspectraltransform = hfft


__all__ = [
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "fftfreq",
    "fftn",
    "ifftn",
    "hfft",
    "fftshift",
    "forwardmultidimensionalfft",
    "inversemultidimensionalfft",
    "hermitianspectraltransform",
]
