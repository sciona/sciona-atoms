from __future__ import annotations

import importlib


def test_scipy_sparse_graph_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.scipy.sparse_graph")
    probes = importlib.import_module("sciona.probes.scipy.sparse_graph")
    assert hasattr(atoms, "graph_laplacian")
    assert hasattr(atoms, "minimum_spanning_tree")
    assert hasattr(probes, "SPARSE_GRAPH_PROBE_TARGETS")
