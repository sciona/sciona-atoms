from __future__ import annotations

from importlib import import_module


def test_namespace_modules_import_cleanly() -> None:
    atoms = import_module("sciona.atoms.signal_processing.biosppy.ecg")
    probes = import_module("sciona.probes.signal_processing.biosppy_ecg")
    assert hasattr(atoms, "heart_rate_computation_median_smoothed")
    assert hasattr(probes, "ECG_PROBE_TARGETS")
