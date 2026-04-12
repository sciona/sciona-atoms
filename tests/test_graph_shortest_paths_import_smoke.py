from __future__ import annotations

import importlib


def test_graph_shortest_paths_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.algorithmic.graph")
    shortest_paths = importlib.import_module("sciona.atoms.algorithmic.graph.shortest_paths")
    probes = importlib.import_module("sciona.probes.algorithmic.graph_shortest_paths")
    assert hasattr(atoms, "dijkstra")
    assert hasattr(atoms, "bellman_ford")
    assert hasattr(atoms, "floyd_warshall")
    assert hasattr(shortest_paths, "dijkstra")
    assert hasattr(shortest_paths, "bellman_ford")
    assert hasattr(shortest_paths, "floyd_warshall")
    assert hasattr(probes, "GRAPH_SHORTEST_PATHS_PROBE_TARGETS")
