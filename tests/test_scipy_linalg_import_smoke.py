from __future__ import annotations

import importlib


def test_scipy_linalg_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.scipy.linalg")
    probes = importlib.import_module("sciona.probes.scipy.linalg")
    assert hasattr(atoms, "solve")
    assert hasattr(atoms, "lu_solve")
    assert hasattr(probes, "LINALG_PROBE_TARGETS")
