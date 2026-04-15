from __future__ import annotations
from typing import Callable, Tuple, Union
import os

import numpy as np
import scipy.signal
import icontract

from sciona.ghost.registry import register_atom
from sciona.ghost.witnesses import (
    witness_butter, witness_cheby1, witness_cheby2, witness_firwin,
    witness_sosfilt, witness_lfilter, witness_freqz,
)

ArrayLike = Union[np.ndarray, list, tuple]

_SLOW_CHECKS = os.environ.get("SCIONA_SLOW_CHECKS", "0") == "1"


def _poles_inside_unit_circle(a: np.ndarray) -> bool:
    """Check that all poles (roots of denominator polynomial) lie inside the unit circle."""
    roots = np.roots(a)
    return bool(np.all(np.abs(roots) < 1.0))


def _is_valid_filter_order(n: int) -> bool:
    """Check that filter order is a positive integer."""
    return isinstance(n, (int, np.integer)) and n > 0


def _is_valid_critical_freq(wn: Union[float, ArrayLike], fs: Union[float, None]) -> bool:
    """Check that critical frequency is in valid range."""
    wn_arr = np.atleast_1d(np.asarray(wn, dtype=float))
    if np.any(wn_arr <= 0):
        return False
    if fs is not None:
        nyquist = fs / 2.0
        if np.any(wn_arr >= nyquist):
            return False
    return True


# ---------------------------------------------------------------------------
# Filter design atoms
# ---------------------------------------------------------------------------

@register_atom(witness_butter)
@icontract.require(lambda N: _is_valid_filter_order(N), "Filter order must be a positive integer")
@icontract.require(lambda Wn, fs: _is_valid_critical_freq(Wn, fs), "Critical frequency must be positive (and < Nyquist if fs given)")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "Output must be (b, a) tuple")
@icontract.ensure(lambda result: result[0].ndim == 1 and result[1].ndim == 1, "b and a must be 1D arrays")
@icontract.ensure(
    lambda result: _poles_inside_unit_circle(result[1]),
    "Designed filter must be stable (poles inside unit circle)",
    enabled=_SLOW_CHECKS,
)
def butter(
    N: int,
    Wn: Union[float, ArrayLike],
    btype: str = "low",
    analog: bool = False,
    output: str = "ba",
    fs: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Design an Nth-order digital or analog Butterworth filter.

    Returns filter coefficients in transfer function (b, a) form.

    Args:
        N: The order of the filter. Must be a positive integer.
        Wn: Critical frequency or frequencies. For digital filters,
            Wn is normalized to [0, 1] where 1 is the Nyquist
            frequency (unless fs is specified).
        btype: Type of filter: 'low', 'high', 'band', or 'stop'.
        analog: If True, return an analog filter.
        output: Type of output: 'ba' for transfer function coefficients.
        fs: The sampling frequency of the digital system.

    Returns:
        Tuple of (b, a) numerator and denominator polynomials.

    """
    b, a = scipy.signal.butter(N, Wn, btype=btype, analog=analog, output=output, fs=fs)
    return b, a


@register_atom(witness_cheby1)
@icontract.require(lambda N: _is_valid_filter_order(N), "Filter order must be a positive integer")
@icontract.require(lambda rp: rp > 0, "Passband ripple must be positive")
@icontract.require(lambda Wn, fs: _is_valid_critical_freq(Wn, fs), "Critical frequency must be positive (and < Nyquist if fs given)")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "Output must be (b, a) tuple")
@icontract.ensure(lambda result: result[0].ndim == 1 and result[1].ndim == 1, "b and a must be 1D arrays")
@icontract.ensure(
    lambda result: _poles_inside_unit_circle(result[1]),
    "Designed filter must be stable (poles inside unit circle)",
    enabled=_SLOW_CHECKS,
)
def cheby1(
    N: int,
    rp: float,
    Wn: Union[float, ArrayLike],
    btype: str = "low",
    analog: bool = False,
    output: str = "ba",
    fs: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Design an Nth-order digital Chebyshev Type I filter.

    Returns filter coefficients with rp decibels of passband ripple.

    Args:
        N: The order of the filter. Must be a positive integer.
        rp: The maximum ripple allowed in the passband, in decibels.
        Wn: Critical frequency or frequencies.
        btype: Type of filter: 'low', 'high', 'band', or 'stop'.
        analog: If True, return an analog filter.
        output: Type of output: 'ba' for transfer function coefficients.
        fs: The sampling frequency of the digital system.

    Returns:
        Tuple of (b, a) numerator and denominator polynomials.

    """
    b, a = scipy.signal.cheby1(N, rp, Wn, btype=btype, analog=analog, output=output, fs=fs)
    return b, a


@register_atom(witness_cheby2)
@icontract.require(lambda N: _is_valid_filter_order(N), "Filter order must be a positive integer")
@icontract.require(lambda rs: rs > 0, "Stopband attenuation must be positive")
@icontract.require(lambda Wn, fs: _is_valid_critical_freq(Wn, fs), "Critical frequency must be positive (and < Nyquist if fs given)")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "Output must be (b, a) tuple")
@icontract.ensure(lambda result: result[0].ndim == 1 and result[1].ndim == 1, "b and a must be 1D arrays")
@icontract.ensure(
    lambda result: _poles_inside_unit_circle(result[1]),
    "Designed filter must be stable (poles inside unit circle)",
    enabled=_SLOW_CHECKS,
)
def cheby2(
    N: int,
    rs: float,
    Wn: Union[float, ArrayLike],
    btype: str = "low",
    analog: bool = False,
    output: str = "ba",
    fs: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Design an Nth-order digital Chebyshev Type II filter.

    Returns filter coefficients with rs decibels of stopband attenuation.

    Args:
        N: The order of the filter. Must be a positive integer.
        rs: The minimum attenuation required in the stop band, in dB.
        Wn: Critical frequency or frequencies.
        btype: Type of filter: 'low', 'high', 'band', or 'stop'.
        analog: If True, return an analog filter.
        output: Type of output: 'ba' for transfer function coefficients.
        fs: The sampling frequency of the digital system.

    Returns:
        Tuple of (b, a) numerator and denominator polynomials.

    """
    b, a = scipy.signal.cheby2(N, rs, Wn, btype=btype, analog=analog, output=output, fs=fs)
    return b, a


# ---------------------------------------------------------------------------
# FIR design
# ---------------------------------------------------------------------------

@register_atom(witness_firwin)
@icontract.require(lambda numtaps: isinstance(numtaps, (int, np.integer)) and numtaps > 0, "numtaps must be a positive integer")
@icontract.ensure(lambda result, numtaps: result.shape == (numtaps,), "Output shape must equal (numtaps,)")
@icontract.ensure(lambda result: np.isrealobj(result), "FIR coefficients must be real-valued")
def firwin(
    numtaps: int,
    cutoff: Union[float, ArrayLike],
    width: float | None = None,
    window: str = "hamming",
    pass_zero: Union[bool, str] = True,
    scale: bool = True,
    fs: float | None = None,
) -> np.ndarray:
    """Design a Finite Impulse Response (FIR) filter using the window method.

    Compute the coefficients of a finite impulse response filter using
    the window method.

    Args:
        numtaps: Length of the filter (number of coefficients). Must
            be odd for Types I and II FIR filters.
        cutoff: Cutoff frequency or frequencies.
        width: Approximate width of transition region.
        window: Window function to use. Default is 'hamming'.
        pass_zero: If True, the DC gain is 1. If False, DC gain is 0.
        scale: If True, scale coefficients so the frequency response
            is exactly unity at certain frequencies.
        fs: The sampling frequency of the signal.

    Returns:
        1D array of FIR filter coefficients with length numtaps.

    """
    return scipy.signal.firwin(
        numtaps, cutoff, width=width, window=window,
        pass_zero=pass_zero, scale=scale, fs=fs,
    )


# ---------------------------------------------------------------------------
# Filter application atoms
# ---------------------------------------------------------------------------

@register_atom(witness_sosfilt)
@icontract.require(lambda sos: np.asarray(sos).ndim == 2 and np.asarray(sos).shape[1] == 6, "sos must have shape (n_sections, 6)")
@icontract.require(lambda x: np.asarray(x).size > 0, "Input signal must not be empty")
@icontract.ensure(lambda result, x: result.shape == np.asarray(x).shape, "Output shape must match input shape")
def sosfilt(
    sos: np.ndarray,
    x: ArrayLike,
    axis: int = -1,
    zi: np.ndarray | None = None,
) -> np.ndarray:
    """Filter data along one dimension using cascaded second-order sections.

    Apply a digital filter in Second-Order Sections (SOS) format to the
    input signal.

    Args:
        sos: Array of second-order filter coefficients with shape
            (n_sections, 6). Each row is [b0, b1, b2, a0, a1, a2].
        x: Input signal array.
        axis: The axis of x to which the filter is applied.
        zi: Initial conditions for the filter delays.

    Returns:
        The filtered output with the same shape as x.

    """
    result = scipy.signal.sosfilt(sos, x, axis=axis, zi=zi)
    if zi is not None:
        return result[0]
    return result


@register_atom(witness_lfilter)
@icontract.require(lambda b: bool(np.asarray(b).ndim == 1), "Numerator b must be 1D")
@icontract.require(lambda a: bool(np.asarray(a).ndim == 1), "Denominator a must be 1D")
@icontract.require(lambda a: np.asarray(a).ndim != 1 or float(np.asarray(a).flat[0]) != 0.0, "Leading denominator coefficient a[0] must not be zero")
@icontract.require(lambda x: np.asarray(x).size > 0, "Input signal must not be empty")
@icontract.ensure(lambda result, x: result.shape == np.asarray(x).shape, "Output shape must match input shape")
def lfilter(
    b: ArrayLike,
    a: ArrayLike,
    x: ArrayLike,
    axis: int = -1,
    zi: np.ndarray | None = None,
) -> np.ndarray:
    """Filter data along one-dimension with an Infinite Impulse Response (IIR) or Finite Impulse Response (FIR) filter.

    Filter a data sequence x using a digital filter described by the
    numerator and denominator coefficient vectors b and a.

    Args:
        b: Numerator coefficient vector of the filter (1D).
        a: Denominator coefficient vector of the filter (1D).
            a[0] must be nonzero.
        x: Input signal array.
        axis: The axis of x to apply the filter.
        zi: Initial conditions for the filter delays.

    Returns:
        The filtered output with the same shape as x.

    """
    result = scipy.signal.lfilter(b, a, x, axis=axis, zi=zi)
    if zi is not None:
        return result[0]
    return result


# ---------------------------------------------------------------------------
# Frequency response
# ---------------------------------------------------------------------------

@register_atom(witness_freqz)
@icontract.require(lambda b: np.asarray(b).size > 0, "Numerator b must be non-empty")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "Output must be (w, h) tuple")
@icontract.ensure(
    lambda result, worN: (
        result[0].shape[0] == (worN if isinstance(worN, int) else len(worN))
        if worN is not None else True
    ),
    "Frequency array length must match worN",
)
def freqz(
    b: ArrayLike,
    a: ArrayLike = 1,
    worN: Union[int, ArrayLike, None] = 512,
    whole: bool = False,
    plot: Callable[..., object] | None = None,
    fs: float = 2 * np.pi,
    include_nyquist: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the frequency response of a digital filter.

    Compute the frequency response of a digital filter given the
    numerator and denominator coefficients.

    Args:
        b: Numerator of the transfer function.
        a: Denominator of the transfer function. Default is 1 (FIR).
        worN: Number of frequencies to compute, or array of
            frequencies. Default is 512.
        whole: If True, compute frequencies from 0 to 2*pi.
        fs: The sampling frequency of the digital system.
            Default is 2*pi (angular frequency).
        include_nyquist: If True, include the Nyquist frequency.

    Returns:
        Tuple of (w, h) where w is the frequency array and h is the
        complex frequency response.

    """
    w, h = scipy.signal.freqz(
        b, a=a, worN=worN, whole=whole, plot=plot, fs=fs,
        include_nyquist=include_nyquist,
    )
    return w, h
