from __future__ import annotations

import importlib


def test_search_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.algorithmic.search")
    probes = importlib.import_module("sciona.probes.algorithmic.search")
    assert hasattr(atoms, "binary_search")
    assert hasattr(atoms, "linear_search")
    assert hasattr(atoms, "hash_lookup")
    assert hasattr(probes, "SEARCH_PROBE_TARGETS")
