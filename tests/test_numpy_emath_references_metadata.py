from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "numpy" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_ATOM_KEYS = {
    "sciona.atoms.numpy.emath.log",
    "sciona.atoms.numpy.emath.log10",
    "sciona.atoms.numpy.emath.logn",
    "sciona.atoms.numpy.emath.power",
    "sciona.atoms.numpy.emath.sqrt",
}


def test_numpy_emath_references_are_canonical_registered_and_manual() -> None:
    refs = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"]

    assert EXPECTED_ATOM_KEYS <= set(refs["atoms"])
    assert registry["numpy_emath_api"]["url"] == "https://numpy.org/doc/stable/reference/routines.emath.html"
    assert registry["numpy_emath_source"]["url"].endswith("/v2.4.2/numpy/lib/_scimath_impl.py")

    for atom_key in EXPECTED_ATOM_KEYS:
        atom_refs = refs["atoms"][atom_key]
        assert {binding["ref_id"] for binding in atom_refs["references"]} == {
            "numpy_emath_api",
            "numpy_emath_source",
        }
        for binding in atom_refs["references"]:
            assert binding["ref_id"] in registry
            metadata = binding["match_metadata"]
            assert metadata["match_type"] == "manual"
            assert metadata["confidence"] == "high"
            assert metadata["notes"]
            assert metadata["matched_nodes"]
