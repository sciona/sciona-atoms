from __future__ import annotations

import importlib


def test_graph_traversal_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.algorithmic.graph")
    traversal = importlib.import_module("sciona.atoms.algorithmic.graph.traversal")
    probes = importlib.import_module("sciona.probes.algorithmic.graph_traversal")
    assert hasattr(atoms, "bfs")
    assert hasattr(atoms, "dfs")
    assert hasattr(traversal, "bfs")
    assert hasattr(traversal, "dfs")
    assert hasattr(probes, "GRAPH_TRAVERSAL_PROBE_TARGETS")
