from __future__ import annotations

import importlib


def test_scipy_spatial_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.scipy.spatial")
    probes = importlib.import_module("sciona.probes.scipy.spatial")
    assert hasattr(atoms, "voronoi_tessellation")
    assert hasattr(atoms, "delaunay_triangulation")
    assert hasattr(probes, "SPATIAL_PROBE_TARGETS")
