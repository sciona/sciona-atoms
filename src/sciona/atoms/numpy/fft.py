"""NumPy FFT atom wrappers for the general provider."""

from __future__ import annotations

import numpy as np
import icontract
from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.registry import register_atom

ArrayLike = np.ndarray | list[object] | tuple[object, ...]


def _shape_from_value(value: AbstractArray | AbstractScalar) -> tuple[int, ...]:
    if isinstance(value, AbstractArray):
        return value.shape
    return ()


def _leading_shape(shape: tuple[int, ...], n: int | None = None) -> tuple[int, ...]:
    if not shape:
        return ()
    tail = shape[-1]
    return shape[:-1] + ((n if n is not None else tail),)


def witness_fft(
    a: AbstractArray | AbstractScalar,
    n: AbstractScalar | None = None,
    axis: AbstractScalar | None = None,
    norm: AbstractScalar | None = None,
) -> AbstractArray | AbstractScalar:
    shape = _shape_from_value(a)
    out_shape = _leading_shape(shape)
    if out_shape == ():
        return AbstractScalar(dtype="complex128")
    return AbstractArray(shape=out_shape, dtype="complex128")


def witness_ifft(
    a: AbstractArray | AbstractScalar,
    n: AbstractScalar | None = None,
    axis: AbstractScalar | None = None,
    norm: AbstractScalar | None = None,
) -> AbstractArray | AbstractScalar:
    return witness_fft(a, n=n, axis=axis, norm=norm)


def witness_rfft(
    a: AbstractArray | AbstractScalar,
    n: AbstractScalar | None = None,
    axis: AbstractScalar | None = None,
    norm: AbstractScalar | None = None,
) -> AbstractArray | AbstractScalar:
    shape = _shape_from_value(a)
    if shape == ():
        return AbstractScalar(dtype="complex128")
    tail = shape[-1]
    out_last = tail // 2 + 1
    return AbstractArray(shape=shape[:-1] + (out_last,), dtype="complex128")


def witness_irfft(
    a: AbstractArray | AbstractScalar,
    n: AbstractScalar | None = None,
    axis: AbstractScalar | None = None,
    norm: AbstractScalar | None = None,
) -> AbstractArray | AbstractScalar:
    shape = _shape_from_value(a)
    if shape == ():
        return AbstractScalar(dtype="float64")
    tail = shape[-1]
    out_last = max(0, 2 * (tail - 1))
    return AbstractArray(shape=shape[:-1] + (out_last,), dtype="float64")


def witness_fftfreq(n: AbstractScalar, d: AbstractScalar | None = None) -> AbstractArray:
    return AbstractArray(shape=(1,), dtype="float64")


def witness_fftn(
    a: AbstractArray | AbstractScalar,
    s: AbstractArray | None = None,
    axes: AbstractArray | None = None,
    norm: AbstractScalar | None = None,
) -> AbstractArray | AbstractScalar:
    return witness_fft(a, axis=None, norm=norm)


def witness_ifftn(
    a: AbstractArray | AbstractScalar,
    s: AbstractArray | None = None,
    axes: AbstractArray | None = None,
    norm: AbstractScalar | None = None,
) -> AbstractArray | AbstractScalar:
    return witness_ifft(a, axis=None, norm=norm)


def witness_hfft(
    a: AbstractArray | AbstractScalar,
    n: AbstractScalar | None = None,
    axis: AbstractScalar | None = None,
    norm: AbstractScalar | None = None,
) -> AbstractArray | AbstractScalar:
    shape = _shape_from_value(a)
    if shape == ():
        return AbstractScalar(dtype="float64")
    out_last = shape[-1]
    return AbstractArray(shape=shape[:-1] + (out_last,), dtype="float64")


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
    lambda result, a, n: result.shape[-1] == (n if n is not None else np.asarray(a).shape[-1]),
    "Result shape must match n or input shape",
)
@icontract.ensure(lambda result: np.iscomplexobj(result), "FFT output must be complex-valued")
def fft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> np.ndarray:
    """Compute the one-dimensional discrete Fourier transform."""
    return np.fft.fft(a, n=n, axis=axis, norm=norm)


@register_atom(witness_ifft)  # type: ignore[untyped-decorator]
@icontract.require(lambda a: a is not None, "Input array must not be None")
@icontract.require(lambda a: np.asarray(a).size > 0, "Input array must not be empty")
@icontract.ensure(
    lambda result, a, n: result.shape[-1] == (n if n is not None else np.asarray(a).shape[-1]),
    "Result shape must match n or input shape",
)
def ifft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> np.ndarray:
    """Compute the inverse one-dimensional discrete Fourier transform."""
    return np.fft.ifft(a, n=n, axis=axis, norm=norm)


@register_atom(witness_rfft)  # type: ignore[untyped-decorator]
@icontract.require(lambda a: a is not None, "Input array must not be None")
@icontract.require(lambda a: np.asarray(a).size > 0, "Input array must not be empty")
@icontract.ensure(
    lambda result, a, n: result.shape[-1] == (n // 2 + 1 if n is not None else np.asarray(a).shape[-1] // 2 + 1),
    "Result shape must match n//2+1 or input shape",
)
@icontract.ensure(lambda result: np.iscomplexobj(result), "RFFT output must be complex-valued")
def rfft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> np.ndarray:
    """Compute the FFT for real-valued input."""
    return np.fft.rfft(a, n=n, axis=axis, norm=norm)


@register_atom(witness_irfft)  # type: ignore[untyped-decorator]
@icontract.require(lambda a: a is not None, "Input array must not be None")
@icontract.require(lambda a: np.asarray(a).size > 0, "Input array must not be empty")
@icontract.ensure(
    lambda result, a, n: result.shape[-1] == (n if n is not None else 2 * (np.asarray(a).shape[-1] - 1)),
    "Result shape must match n or inferred input shape",
)
@icontract.ensure(lambda result: np.isrealobj(result), "IRFFT output must be real-valued")
def irfft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> np.ndarray:
    """Compute the inverse FFT for a real-input spectrum."""
    return np.fft.irfft(a, n=n, axis=axis, norm=norm)


@register_atom(witness_fftfreq)  # type: ignore[untyped-decorator]
@icontract.require(lambda n: n > 0, "n must be positive")
@icontract.require(lambda d: d > 0, "d must be positive")
@icontract.ensure(lambda result, n: result.shape == (n,), "Result shape must match n")
def fftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """Return the discrete Fourier transform sample frequencies."""
    return np.fft.fftfreq(n, d=d)


@register_atom(witness_fftn)  # type: ignore[untyped-decorator]
@icontract.require(lambda a: a is not None, "Input array must not be None")
@icontract.require(lambda a: np.asarray(a).size > 0, "Input array must not be empty")
@icontract.ensure(lambda result: np.iscomplexobj(result), "FFTN output must be complex-valued")
def fftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> np.ndarray:
    """Compute the N-dimensional discrete Fourier transform."""
    return np.fft.fftn(a, s=s, axes=axes, norm=norm)


@register_atom(witness_ifftn)  # type: ignore[untyped-decorator]
@icontract.require(lambda a: a is not None, "Input array must not be None")
@icontract.require(lambda a: np.asarray(a).size > 0, "Input array must not be empty")
@icontract.ensure(lambda result: np.iscomplexobj(result), "IFFTN output must be complex-valued")
def ifftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> np.ndarray:
    """Compute the inverse N-dimensional discrete Fourier transform."""
    return np.fft.ifftn(a, s=s, axes=axes, norm=norm)


@register_atom(witness_hfft)  # type: ignore[untyped-decorator]
@icontract.require(lambda a: a is not None, "Input array must not be None")
@icontract.ensure(lambda result: result is not None, "HFFT output must not be None")
def hfft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> np.ndarray:
    """Compute the FFT of a signal with Hermitian symmetry."""
    return np.fft.hfft(a, n=n, axis=axis, norm=norm)


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
