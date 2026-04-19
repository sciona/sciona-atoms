from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "numpy" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_ATOM_KEYS = {
    "sciona.atoms.numpy.arrays.array",
    "sciona.atoms.numpy.arrays.dot",
    "sciona.atoms.numpy.arrays.reshape",
    "sciona.atoms.numpy.arrays.vstack",
    "sciona.atoms.numpy.arrays.zeros",
}


def test_numpy_arrays_references_are_canonical_registered_and_manual() -> None:
    refs = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"]

    assert EXPECTED_ATOM_KEYS <= set(refs["atoms"])
    assert registry["numpy_arrays_api"]["url"] == "https://numpy.org/doc/stable/reference/routines.html"

    for atom_key in EXPECTED_ATOM_KEYS:
        atom_refs = refs["atoms"][atom_key]
        assert atom_refs["references"]
        for binding in atom_refs["references"]:
            assert binding["ref_id"] in registry
            metadata = binding["match_metadata"]
            assert metadata["match_type"] == "manual"
            assert metadata["confidence"] == "high"
            assert metadata["notes"]
            assert metadata["matched_nodes"]
