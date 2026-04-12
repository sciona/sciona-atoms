from __future__ import annotations

import importlib


def test_numpy_search_sort_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.numpy.search_sort")
    probes = importlib.import_module("sciona.probes.numpy.search_sort")
    assert hasattr(atoms, "binary_search_insertion")
    assert hasattr(atoms, "lexicographic_indirect_sort")
    assert hasattr(atoms, "partial_sort_partition")
    assert hasattr(probes, "SEARCH_SORT_PROBE_TARGETS")
