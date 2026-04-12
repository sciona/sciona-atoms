from __future__ import annotations

import importlib


def test_scipy_integrate_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.scipy.integrate")
    probes = importlib.import_module("sciona.probes.scipy.integrate")
    assert hasattr(atoms, "quad")
    assert hasattr(atoms, "solve_ivp")
    assert hasattr(atoms, "simpson")
    assert hasattr(probes, "INTEGRATE_PROBE_TARGETS")
