from __future__ import annotations

import json
from pathlib import Path

from sciona.atoms.audit_review_bundles import load_review_bundle_entries


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "numpy_arrays_pubrev_030.review_bundle.json"

EXPECTED_ATOM_KEYS = {
    "sciona.atoms.numpy.arrays.array",
    "sciona.atoms.numpy.arrays.dot",
    "sciona.atoms.numpy.arrays.reshape",
    "sciona.atoms.numpy.arrays.vstack",
    "sciona.atoms.numpy.arrays.zeros",
}

EXPECTED_SOURCE_PATHS = {
    "src/sciona/atoms/numpy/arrays.py",
    "src/sciona/probes/numpy/arrays.py",
    "src/sciona/atoms/numpy/references.json",
    "tests/test_numpy_arrays_behavior.py",
    "tests/test_numpy_arrays_references_metadata.py",
    "tests/test_numpy_arrays_review_bundle.py",
}


def test_numpy_arrays_pubrev_030_review_bundle_is_publishable_with_limits() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))

    assert bundle["provider_repo"] == "sciona-atoms"
    assert bundle["family_batch"] == "numpy_arrays_pubrev_030"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass"
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] == "catalog_ready"
    assert bundle["blocking_findings"] == []
    assert bundle["required_actions"] == []

    rows = bundle["rows"]
    assert {row["atom_key"] for row in rows} == EXPECTED_ATOM_KEYS

    limited_atoms = {
        "sciona.atoms.numpy.arrays.array",
        "sciona.atoms.numpy.arrays.zeros",
        "sciona.atoms.numpy.arrays.vstack",
    }
    for row in rows:
        assert row["atom_name"] == row["atom_key"]
        assert row["review_status"] == "reviewed"
        assert row["review_semantic_verdict"] == "pass"
        assert row["review_developer_semantic_verdict"] == "pass_with_limits"
        assert row["trust_readiness"] == "catalog_ready"
        assert row["overall_verdict"] == "acceptable_with_limits"
        assert row["structural_status"] == "pass"
        assert row["semantic_status"] == "pass"
        assert row["runtime_status"] == "pass"
        assert row["developer_semantics_status"] == "pass_with_limits"
        assert row["risk_tier"] == "low"
        assert isinstance(row["risk_score"], int)
        assert row["acceptability_band"] == "review_ready"
        assert row["parity_test_status"] == "pass"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["review_record_path"] == "data/review_bundles/numpy_arrays_pubrev_030.review_bundle.json"
        assert set(row["source_paths"]) == EXPECTED_SOURCE_PATHS
        for rel_path in row["source_paths"]:
            assert (ROOT / rel_path).exists()
        if row["atom_key"] in limited_atoms:
            assert row["limitations"]


def test_numpy_arrays_pubrev_030_bundle_merges_as_approved_rows() -> None:
    entries = load_review_bundle_entries(BUNDLE_PATH)

    assert {entry.atom_name for entry in entries} == EXPECTED_ATOM_KEYS
    for entry in entries:
        assert entry.patch["review_status"] == "approved"
        assert entry.patch["review_priority"] == "review_now"
        assert entry.patch["trust_readiness"] == "reviewed_with_limits"
        assert entry.patch["review_semantic_verdict"] == "pass"
        assert entry.patch["review_developer_semantics_verdict"] == "pass_with_limits"
        assert entry.patch["overall_verdict"] == "acceptable_with_limits"
        assert entry.patch["blocking_findings"] == []
        assert entry.patch["required_actions"] == []
