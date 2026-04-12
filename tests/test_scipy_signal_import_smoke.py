from __future__ import annotations

import importlib


def test_scipy_signal_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.scipy.signal")
    probes = importlib.import_module("sciona.probes.scipy.signal")
    assert hasattr(atoms, "butter")
    assert hasattr(atoms, "freqz")
    assert hasattr(probes, "SIGNAL_PROBE_TARGETS")
