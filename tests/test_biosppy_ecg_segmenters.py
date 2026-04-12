from __future__ import annotations

import numpy as np

from sciona.atoms.signal_processing.biosppy import ecg as ecg_atoms


def test_ssf_segmenter_returns_first_plausible_output(monkeypatch) -> None:
    calls: list[float] = []

    def _fake_ssf_segmenter(*, signal: np.ndarray, sampling_rate: float, threshold: float):
        calls.append(float(threshold))
        return {"rpeaks": np.array([100, 900, 1700, 2500, 3300, 4100])}

    def _unexpected_hamilton(*, signal: np.ndarray, sampling_rate: float):
        raise AssertionError("Hamilton fallback should not run when SSF is plausible")

    monkeypatch.setattr(ecg_atoms.biosppy_ecg, "ssf_segmenter", _fake_ssf_segmenter)
    monkeypatch.setattr(ecg_atoms.biosppy_ecg, "hamilton_segmenter", _unexpected_hamilton)

    result = ecg_atoms.ssf_segmenter(np.zeros(5000), sampling_rate=1000.0)

    assert calls == [20.0]
    assert np.array_equal(result, np.array([100, 900, 1700, 2500, 3300, 4100]))


def test_ssf_segmenter_falls_back_to_hamilton_when_no_candidate_is_plausible(
    monkeypatch,
) -> None:
    calls: list[float] = []

    def _fake_ssf_segmenter(*, signal: np.ndarray, sampling_rate: float, threshold: float):
        calls.append(float(threshold))
        return {"rpeaks": np.array([100, 150, 200])}

    def _fake_hamilton(*, signal: np.ndarray, sampling_rate: float):
        return {"rpeaks": np.array([120, 920, 1720, 2520, 3320, 4120])}

    monkeypatch.setattr(ecg_atoms.biosppy_ecg, "ssf_segmenter", _fake_ssf_segmenter)
    monkeypatch.setattr(ecg_atoms.biosppy_ecg, "hamilton_segmenter", _fake_hamilton)

    result = ecg_atoms.ssf_segmenter(np.zeros(5000), sampling_rate=1000.0)

    assert calls == [20.0, 5.0, 1.0, 0.2, 0.1, 0.05]
    assert np.array_equal(result, np.array([120, 920, 1720, 2520, 3320, 4120]))


def test_christov_segmenter_extracts_rpeaks_from_mapping_result(monkeypatch) -> None:
    def _fake_christov(*, signal: np.ndarray, sampling_rate: float):
        return {"rpeaks": np.array([50, 450, 850])}

    monkeypatch.setattr(ecg_atoms.biosppy_ecg, "christov_segmenter", _fake_christov)

    result = ecg_atoms.christov_segmenter(np.zeros(1000), sampling_rate=1000.0)

    assert np.array_equal(result, np.array([50, 450, 850]))
