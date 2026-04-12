from __future__ import annotations

import numpy as np

from sciona.atoms.signal_processing.biosppy.ecg import (
    bandpass_filter,
    heart_rate_computation,
    heart_rate_computation_median_smoothed,
    peak_correction,
    reject_outlier_intervals,
    template_extraction,
)


def _synthetic_ecg_signal(length: int, peak_indices: list[int]) -> np.ndarray:
    signal = np.zeros(length, dtype=float)
    for peak in peak_indices:
        if 2 <= peak < length - 2:
            signal[peak - 2 : peak + 3] += np.array([0.1, 0.5, 1.2, 0.5, 0.1])
    return signal


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
    assert 2660 not in cleaned


def test_median_smoothed_rate_returns_finite_values() -> None:
    rpeaks = np.array([100, 950, 1800, 2660, 3490, 4350, 5200])
    indices, rate = heart_rate_computation_median_smoothed(rpeaks, sampling_rate=1000.0)
    assert indices.shape == rate.shape
    assert rate.size > 0
    assert np.isfinite(rate).all()


def test_peak_correction_and_template_extraction_return_aligned_outputs() -> None:
    signal = _synthetic_ecg_signal(1200, [200, 500, 800, 1100])
    initial_rpeaks = np.array([198, 498, 803, 1098])

    corrected = peak_correction(signal, initial_rpeaks, sampling_rate=1000.0, tol=0.01)
    templates, template_rpeaks = template_extraction(signal, corrected, sampling_rate=1000.0)

    assert corrected.shape == initial_rpeaks.shape
    assert np.max(np.abs(corrected - np.array([200, 500, 800, 1100]))) <= 1
    assert templates.shape[0] == template_rpeaks.shape[0]
    assert templates.shape[0] >= 3
    assert np.isfinite(templates).all()


def test_heart_rate_computation_window_one_matches_unsmoothed() -> None:
    rpeaks = np.array([100, 900, 1700, 2500, 3300, 4100])

    raw_indices, raw_rate = heart_rate_computation(rpeaks, sampling_rate=1000.0)
    smooth_indices, smooth_rate = heart_rate_computation_median_smoothed(
        rpeaks,
        sampling_rate=1000.0,
        smoothing_window=1,
    )

    assert np.array_equal(raw_indices, smooth_indices)
    assert np.allclose(raw_rate, smooth_rate)
