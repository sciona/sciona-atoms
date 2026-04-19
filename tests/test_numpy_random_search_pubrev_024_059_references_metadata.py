from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "numpy" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_ATOMS = {
    "sciona.atoms.numpy.random.combinatorics_sampler",
    "sciona.atoms.numpy.random.continuous_multivariate_sampler",
    "sciona.atoms.numpy.random.default_rng",
    "sciona.atoms.numpy.random.discrete_event_sampler",
    "sciona.atoms.numpy.random.rand",
    "sciona.atoms.numpy.random.uniform",
    "sciona.atoms.numpy.search_sort.binary_search_insertion",
    "sciona.atoms.numpy.search_sort.lexicographic_indirect_sort",
    "sciona.atoms.numpy.search_sort.partial_sort_partition",
}


def test_numpy_random_search_references_are_repo_local_and_complete() -> None:
    references = json.loads(REFERENCES_PATH.read_text())
    registry = json.loads(REGISTRY_PATH.read_text())
    registry_ids = set(registry["references"])

    assert EXPECTED_ATOMS <= set(references["atoms"])

    for atom_key in EXPECTED_ATOMS:
        atom_refs = references["atoms"][atom_key]["references"]
        assert atom_refs
        for ref in atom_refs:
            assert ref["ref_id"] in registry_ids
            metadata = ref["match_metadata"]
            assert metadata["match_type"]
            assert metadata["confidence"] in {"high", "medium", "low"}
            assert metadata["notes"]
