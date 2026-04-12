from __future__ import annotations

import importlib


def test_numpy_linalg_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.numpy.linalg")
    probes = importlib.import_module("sciona.probes.numpy.linalg")
    assert hasattr(atoms, "solve")
    assert hasattr(atoms, "inv")
    assert hasattr(atoms, "det")
    assert hasattr(atoms, "norm")
    assert hasattr(probes, "LINALG_PROBE_TARGETS")
