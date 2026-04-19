from __future__ import annotations

import json
from pathlib import Path

from sciona.atoms.audit_review_bundles import load_review_bundle_entries
from sciona.atoms.expansion.sequential_filter import SEQUENTIAL_FILTER_DECLARATIONS
from sciona.probes.expansion.sequential_filter import SEQUENTIAL_FILTER_PROBE_TARGETS


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "sequential_filter_pubrev_044.review_bundle.json"

EXPECTED_SOURCE_PATHS = {
    "src/sciona/atoms/expansion/sequential_filter.py",
    "src/sciona/probes/expansion/sequential_filter.py",
    "src/sciona/atoms/expansion/sequential_filter/references.json",
    "data/references/registry.json",
    "tests/test_expansion_sequential_filter.py",
    "tests/test_expansion_sequential_filter_references_metadata.py",
    "tests/test_expansion_sequential_filter_review_bundle.py",
}


def test_sequential_filter_pubrev_044_review_bundle_is_publishable_with_limits() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))

    declaration_fqdns = {fqdn for fqdn, _, _ in SEQUENTIAL_FILTER_DECLARATIONS.values()}
    probe_fqdns = {target.atom_fqdn for target in SEQUENTIAL_FILTER_PROBE_TARGETS}

    assert bundle["provider_repo"] == "sciona-atoms"
    assert bundle["family_batch"] == "sequential_filter_pubrev_044"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass"
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] == "catalog_ready"
    assert bundle["blocking_findings"] == []
    assert bundle["required_actions"] == []
    assert declaration_fqdns == probe_fqdns

    authoritative_paths = {source["path"] for source in bundle["authoritative_sources"]}
    assert EXPECTED_SOURCE_PATHS < authoritative_paths
    assert "tests/test_expansion_registration_import_smoke.py" in authoritative_paths

    rows = bundle["rows"]
    assert {row["atom_key"] for row in rows} == declaration_fqdns
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
        assert row["review_record_path"] == (
            "data/review_bundles/sequential_filter_pubrev_044.review_bundle.json"
        )
        assert set(row["source_paths"]) == EXPECTED_SOURCE_PATHS
        assert row["limitations"]
        for rel_path in row["source_paths"]:
            assert (ROOT / rel_path).exists()


def test_sequential_filter_pubrev_044_bundle_merges_as_approved_rows() -> None:
    entries = load_review_bundle_entries(BUNDLE_PATH)
    expected_fqdns = {fqdn for fqdn, _, _ in SEQUENTIAL_FILTER_DECLARATIONS.values()}

    assert {entry.atom_name for entry in entries} == expected_fqdns
    for entry in entries:
        assert entry.patch["review_status"] == "approved"
        assert entry.patch["review_priority"] == "review_now"
        assert entry.patch["trust_readiness"] == "reviewed_with_limits"
        assert entry.patch["review_semantic_verdict"] == "pass"
        assert entry.patch["review_developer_semantics_verdict"] == "pass_with_limits"
        assert entry.patch["overall_verdict"] == "acceptable_with_limits"
        assert entry.patch["blocking_findings"] == []
        assert entry.patch["required_actions"] == []
