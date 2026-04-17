from __future__ import annotations

import importlib


def test_scipy_interpolate_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.scipy.interpolate")
    probes = importlib.import_module("sciona.probes.scipy.interpolate")
    assert hasattr(atoms, "cubic_spline_fit")
    assert hasattr(atoms, "rbf_interpolator_fit")
    assert hasattr(probes, "INTERPOLATE_PROBE_TARGETS")
