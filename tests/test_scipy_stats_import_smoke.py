from __future__ import annotations

import importlib


def test_scipy_stats_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.scipy.stats")
    probes = importlib.import_module("sciona.probes.scipy.stats")
    assert hasattr(atoms, "describe")
    assert hasattr(atoms, "norm")
    assert hasattr(probes, "STATS_PROBE_TARGETS")
