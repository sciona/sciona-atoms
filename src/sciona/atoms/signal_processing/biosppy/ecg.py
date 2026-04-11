"""BioSPPy ECG atom wrappers for the signal-processing namespace pilot."""

from __future__ import annotations

from typing import Any

import biosppy.signals.ecg as biosppy_ecg
import biosppy.signals.tools as biosppy_tools
import icontract
import numpy as np


def _is_vector(array: np.ndarray) -> bool:
    return isinstance(array, np.ndarray) and array.ndim == 1


def _valid_sampling_rate(sampling_rate: float) -> bool:
    return isinstance(sampling_rate, (float, int, np.number)) and float(sampling_rate) > 0.0


def _extract_rpeaks(result: Any) -> np.ndarray:
    if isinstance(result, dict):
        return np.asarray(result["rpeaks"], dtype=int)
    return np.asarray(result[0], dtype=int)


def _rr_irregularity(rpeaks: np.ndarray) -> float:
    if len(rpeaks) < 3:
        return 0.0
    rr = np.diff(rpeaks)
    mean_rr = float(np.mean(rr))
    if mean_rr <= 0.0:
        return float("inf")
    return float(np.std(rr) / mean_rr)


def _mean_heart_rate_bpm(rpeaks: np.ndarray, sampling_rate: float) -> float:
    if len(rpeaks) < 2:
        return float("nan")
    rr = np.diff(rpeaks) / float(sampling_rate)
    mean_rr = float(np.mean(rr))
    if mean_rr <= 0.0:
        return float("nan")
    return 60.0 / mean_rr


def _plausible_segmenter_output(rpeaks: np.ndarray, sampling_rate: float) -> bool:
    mean_hr = _mean_heart_rate_bpm(rpeaks, sampling_rate)
    return len(rpeaks) >= 2 and 40.0 <= mean_hr <= 200.0 and _rr_irregularity(rpeaks) <= 0.25


@icontract.require(lambda signal: _is_vector(signal), "signal must be a 1D numpy array")
@icontract.require(_valid_sampling_rate, "sampling_rate must be positive")
@icontract.ensure(lambda result: result is not None, "Bandpass Filter output must not be None")
def bandpass_filter(signal: np.ndarray, *, sampling_rate: float = 1000.0) -> np.ndarray:
    """Apply FIR bandpass filtering to an ECG waveform."""
    order = int(0.3 * float(sampling_rate))
    filtered, _, _ = biosppy_tools.filter_signal(
        signal=signal,
        ftype="FIR",
        band="bandpass",
        order=order,
        frequency=[3, 45],
        sampling_rate=float(sampling_rate),
    )
    return filtered


@icontract.require(lambda filtered: _is_vector(filtered), "filtered must be a 1D numpy array")
@icontract.require(_valid_sampling_rate, "sampling_rate must be positive")
@icontract.ensure(lambda result: result is not None, "R-Peak Detection output must not be None")
def r_peak_detection(filtered: np.ndarray, *, sampling_rate: float = 1000.0) -> np.ndarray:
    """Detect R-peak sample indices from a filtered ECG signal."""
    return biosppy_ecg.hamilton_segmenter(signal=filtered, sampling_rate=float(sampling_rate))["rpeaks"]


@icontract.require(lambda signal: _is_vector(signal), "signal must be a 1D numpy array")
@icontract.require(lambda rpeaks: _is_vector(rpeaks), "rpeaks must be a 1D numpy array")
@icontract.require(_valid_sampling_rate, "sampling_rate must be positive")
@icontract.ensure(lambda result: result is not None, "Peak Correction output must not be None")
def peak_correction(
    signal: np.ndarray,
    rpeaks: np.ndarray,
    *,
    sampling_rate: float = 1000.0,
    tol: float = 0.05,
) -> np.ndarray:
    """Correct candidate R-peak locations against an ECG signal."""
    return biosppy_ecg.correct_rpeaks(
        signal=signal,
        rpeaks=rpeaks,
        sampling_rate=float(sampling_rate),
        tol=float(tol),
    )["rpeaks"]


@icontract.require(lambda rpeaks: _is_vector(rpeaks), "rpeaks must be a 1D numpy array")
@icontract.require(_valid_sampling_rate, "sampling_rate must be positive")
@icontract.require(
    lambda mad_scale: isinstance(mad_scale, (float, int, np.number)) and float(mad_scale) > 0.0,
    "mad_scale must be positive",
)
@icontract.ensure(lambda result: result is not None, "Outlier Interval Rejection output must not be None")
def reject_outlier_intervals(
    rpeaks: np.ndarray,
    *,
    sampling_rate: float = 1000.0,
    mad_scale: float = 3.0,
    min_interval_s: float = 0.25,
    max_interval_s: float = 2.0,
) -> np.ndarray:
    """Remove event markers that induce implausible adjacent intervals."""
    cleaned = np.asarray(rpeaks, dtype=int).reshape(-1)
    if cleaned.size < 5:
        return cleaned

    sr = float(sampling_rate)
    if sr <= 0.0:
        return cleaned

    rr = np.diff(cleaned).astype(float)
    if rr.size < 3:
        return cleaned

    median_rr = float(np.median(rr))
    mad_rr = float(np.median(np.abs(rr - median_rr)))
    if mad_rr <= 1e-9:
        return cleaned

    lo = max(float(min_interval_s) * sr, median_rr - float(mad_scale) * mad_rr)
    hi = min(float(max_interval_s) * sr, median_rr + float(mad_scale) * mad_rr)

    keep = np.ones(cleaned.size, dtype=bool)
    for interval_index, interval in enumerate(rr):
        if interval < lo or interval > hi:
            later_event_index = interval_index + 1
            if later_event_index < cleaned.size - 1:
                keep[later_event_index] = False

    filtered = cleaned[keep]
    if filtered.size >= 3:
        return filtered.astype(int, copy=False)
    return cleaned.astype(int, copy=False)


@icontract.require(lambda signal: _is_vector(signal), "signal must be a 1D numpy array")
@icontract.require(lambda rpeaks: _is_vector(rpeaks), "rpeaks must be a 1D numpy array")
@icontract.require(_valid_sampling_rate, "sampling_rate must be positive")
@icontract.ensure(lambda result: all(item is not None for item in result), "Template Extraction outputs must not be None")
def template_extraction(
    signal: np.ndarray,
    rpeaks: np.ndarray,
    *,
    sampling_rate: float = 1000.0,
    before: float = 0.2,
    after: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract heartbeat templates around corrected R-peaks."""
    result = biosppy_ecg.extract_heartbeats(
        signal=signal,
        rpeaks=rpeaks,
        sampling_rate=float(sampling_rate),
        before=float(before),
        after=float(after),
    )
    return result["templates"], result["rpeaks"]


@icontract.require(lambda rpeaks: _is_vector(rpeaks), "rpeaks must be a 1D numpy array")
@icontract.require(_valid_sampling_rate, "sampling_rate must be positive")
@icontract.ensure(lambda result: all(item is not None for item in result), "Heart Rate Computation outputs must not be None")
def heart_rate_computation(rpeaks: np.ndarray, *, sampling_rate: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute instantaneous heart rate from R-peak indices."""
    result = biosppy_tools.get_heart_rate(beats=rpeaks, sampling_rate=float(sampling_rate), smooth=False)
    return result["index"], result["heart_rate"]


@icontract.require(lambda rpeaks: _is_vector(rpeaks), "rpeaks must be a 1D numpy array")
@icontract.require(_valid_sampling_rate, "sampling_rate must be positive")
@icontract.ensure(lambda result: all(item is not None for item in result), "Robust Smoothed Heart Rate Computation outputs must not be None")
def heart_rate_computation_median_smoothed(
    rpeaks: np.ndarray,
    *,
    sampling_rate: float = 1000.0,
    smoothing_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute instantaneous heart rate and apply robust median smoothing."""
    indices, heart_rate = heart_rate_computation(rpeaks, sampling_rate=float(sampling_rate))
    rate = np.asarray(heart_rate, dtype=float).reshape(-1)
    if rate.size == 0:
        return np.asarray(indices, dtype=int), rate

    window = max(1, int(smoothing_window))
    if window > rate.size:
        window = int(rate.size)
    if window % 2 == 0:
        window = max(1, window - 1)
    if window <= 1:
        return np.asarray(indices, dtype=int), rate

    padded = np.pad(rate, (window // 2, window // 2), mode="edge")
    smoothed = np.empty_like(rate)
    for idx in range(rate.size):
        smoothed[idx] = float(np.median(padded[idx : idx + window]))
    return np.asarray(indices, dtype=int), smoothed


@icontract.require(lambda signal: _is_vector(signal), "signal must be a 1D numpy array")
@icontract.require(_valid_sampling_rate, "sampling_rate must be positive")
@icontract.ensure(lambda result: result is not None, "SSF Segmenter output must not be None")
def ssf_segmenter(signal: np.ndarray, *, sampling_rate: float = 1000.0) -> np.ndarray:
    """Detect ECG peaks with the slope-sum-function segmenter."""
    thresholds = (20.0, 5.0, 1.0, 0.2, 0.1, 0.05)
    best = np.array([], dtype=int)
    for threshold in thresholds:
        result = biosppy_ecg.ssf_segmenter(signal=signal, sampling_rate=float(sampling_rate), threshold=threshold)
        rpeaks = _extract_rpeaks(result)
        if len(rpeaks) > len(best):
            best = rpeaks
        if _plausible_segmenter_output(rpeaks, sampling_rate):
            return rpeaks
    if _plausible_segmenter_output(best, sampling_rate):
        return best
    return _extract_rpeaks(biosppy_ecg.hamilton_segmenter(signal=signal, sampling_rate=float(sampling_rate)))


@icontract.require(lambda signal: _is_vector(signal), "signal must be a 1D numpy array")
@icontract.require(_valid_sampling_rate, "sampling_rate must be positive")
@icontract.ensure(lambda result: result is not None, "Christov Segmenter output must not be None")
def christov_segmenter(signal: np.ndarray, *, sampling_rate: float = 1000.0) -> np.ndarray:
    """Detect ECG peaks with the Christov segmenter."""
    result = biosppy_ecg.christov_segmenter(signal=signal, sampling_rate=float(sampling_rate))
    return _extract_rpeaks(result)
