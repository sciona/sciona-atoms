from __future__ import annotations

import importlib


def test_numpy_emath_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.numpy.emath")
    probes = importlib.import_module("sciona.probes.numpy.emath")
    assert hasattr(atoms, "sqrt")
    assert hasattr(atoms, "log")
    assert hasattr(atoms, "log10")
    assert hasattr(atoms, "logn")
    assert hasattr(atoms, "power")
    assert hasattr(probes, "EMATH_PROBE_TARGETS")
