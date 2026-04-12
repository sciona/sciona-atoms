from __future__ import annotations

import importlib


def test_numpy_arrays_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.numpy.arrays")
    probes = importlib.import_module("sciona.probes.numpy.arrays")
    assert hasattr(atoms, "array")
    assert hasattr(atoms, "zeros")
    assert hasattr(atoms, "dot")
    assert hasattr(atoms, "vstack")
    assert hasattr(atoms, "reshape")
    assert hasattr(probes, "ARRAYS_PROBE_TARGETS")
