from __future__ import annotations

import importlib


def test_numpy_polynomial_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.numpy.polynomial")
    probes = importlib.import_module("sciona.probes.numpy.polynomial")
    assert hasattr(atoms, "polyval")
    assert hasattr(atoms, "polyfit")
    assert hasattr(atoms, "polyder")
    assert hasattr(atoms, "polyint")
    assert hasattr(atoms, "polyadd")
    assert hasattr(atoms, "polymul")
    assert hasattr(atoms, "polyroots")
    assert hasattr(probes, "POLYNOMIAL_PROBE_TARGETS")
