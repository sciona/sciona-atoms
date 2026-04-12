from __future__ import annotations

import importlib


def test_numpy_fft_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.numpy.fft")
    probes = importlib.import_module("sciona.probes.numpy.fft")
    assert hasattr(atoms, "fft")
    assert hasattr(atoms, "ifft")
    assert hasattr(atoms, "rfft")
    assert hasattr(atoms, "irfft")
    assert hasattr(atoms, "fftfreq")
    assert hasattr(atoms, "fftn")
    assert hasattr(atoms, "ifftn")
    assert hasattr(atoms, "hfft")
    assert hasattr(atoms, "fftshift")
    assert hasattr(probes, "FFT_PROBE_TARGETS")
