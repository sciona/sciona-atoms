from __future__ import annotations

import importlib


def test_scipy_fft_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.scipy.fft")
    probes = importlib.import_module("sciona.probes.scipy.fft")
    assert hasattr(atoms, "dct")
    assert hasattr(atoms, "idct")
    assert hasattr(probes, "FFT_PROBE_TARGETS")
