from __future__ import annotations

import importlib


def test_scipy_optimize_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.scipy.optimize")
    probes = importlib.import_module("sciona.probes.scipy.optimize")
    assert hasattr(atoms, "minimize")
    assert hasattr(atoms, "shgo")
    assert hasattr(atoms, "differential_evolution")
    assert hasattr(probes, "OPTIMIZE_PROBE_TARGETS")
