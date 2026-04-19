from __future__ import annotations

import json
from pathlib import Path

from sciona.atoms.audit_review_bundles import load_review_bundle_entries


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "numpy_emath_pubrev_031.review_bundle.json"

EXPECTED_ATOM_KEYS = {
    "sciona.atoms.numpy.emath.log",
    "sciona.atoms.numpy.emath.log10",
    "sciona.atoms.numpy.emath.logn",
    "sciona.atoms.numpy.emath.power",
    "sciona.atoms.numpy.emath.sqrt",
}

EXPECTED_SOURCE_PATHS = {
    "src/sciona/atoms/numpy/emath.py",
    "src/sciona/probes/numpy/emath.py",
    "src/sciona/atoms/numpy/witnesses.py",
    "src/sciona/atoms/numpy/references.json",
    "data/references/registry.json",
    "tests/test_numpy_emath_behavior.py",
    "tests/test_numpy_emath_references_metadata.py",
    "tests/test_numpy_emath_review_bundle.py",
}


def test_numpy_emath_pubrev_031_review_bundle_is_publishable_with_limits() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))

    assert bundle["provider_repo"] == "sciona-atoms"
    assert bundle["family_batch"] == "numpy_emath_pubrev_031"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass"
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] == "catalog_ready"
    assert bundle["blocking_findings"] == []
    assert bundle["required_actions"] == []
    assert any(source["kind"] == "upstream_source" for source in bundle["authoritative_sources"])

    rows = bundle["rows"]
    assert {row["atom_key"] for row in rows} == EXPECTED_ATOM_KEYS
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
        assert row["review_record_path"] == "data/review_bundles/numpy_emath_pubrev_031.review_bundle.json"
        assert set(row["source_paths"]) == EXPECTED_SOURCE_PATHS
        assert row["limitations"]
        for rel_path in row["source_paths"]:
            assert (ROOT / rel_path).exists()


def test_numpy_emath_pubrev_031_bundle_merges_as_approved_rows() -> None:
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
