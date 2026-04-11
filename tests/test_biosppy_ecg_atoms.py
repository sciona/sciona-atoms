from __future__ import annotations

import numpy as np

from sciona.atoms.signal_processing.biosppy.ecg import (
    bandpass_filter,
    heart_rate_computation_median_smoothed,
    reject_outlier_intervals,
)


def test_bandpass_filter_preserves_shape() -> None:
    sampling_rate = 1000.0
    t = np.linspace(0.0, 2.0, int(2.0 * sampling_rate), endpoint=False)
    signal = np.sin(2 * np.pi * 8.0 * t) + 0.1 * np.sin(2 * np.pi * 40.0 * t)
    filtered = bandpass_filter(signal, sampling_rate=sampling_rate)
    assert filtered.shape == signal.shape
    assert np.isfinite(filtered).all()


def test_reject_outlier_intervals_removes_implausible_rr_jump() -> None:
    rpeaks = np.array([100, 900, 1730, 2550, 2660, 3490, 4310])
    cleaned = reject_outlier_intervals(rpeaks, sampling_rate=1000.0)
    assert len(cleaned) < len(rpeaks)
    assert 2600 not in cleaned


def test_median_smoothed_rate_returns_finite_values() -> None:
    rpeaks = np.array([100, 950, 1800, 2660, 3490, 4350, 5200])
    indices, rate = heart_rate_computation_median_smoothed(rpeaks, sampling_rate=1000.0)
    assert indices.shape == rate.shape
    assert rate.size > 0
    assert np.isfinite(rate).all()
