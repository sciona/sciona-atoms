from __future__ import annotations

import json
from pathlib import Path

from sciona.atoms.expansion.divide_and_conquer import DIVIDE_AND_CONQUER_DECLARATIONS
from sciona.probes.expansion.divide_and_conquer import DIVIDE_AND_CONQUER_PROBE_TARGETS


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "divide_and_conquer_expansion.review_bundle.json"

EXPECTED_AUTHORITATIVE_SOURCES = {
    ("local_source", "src/sciona/atoms/expansion/divide_and_conquer.py"),
    ("local_source", "src/sciona/atoms/expansion/divide_and_conquer/references.json"),
    ("local_source", "src/sciona/probes/expansion/divide_and_conquer.py"),
    ("local_test", "tests/test_expansion_divide_and_conquer.py"),
    ("local_test", "tests/test_divide_and_conquer_references_metadata.py"),
    ("local_test", "tests/test_expansion_import_smoke.py"),
    ("local_test", "tests/test_expansion_registration_import_smoke.py"),
}
EXPECTED_ROW_SOURCE_PATHS = {
    "src/sciona/atoms/expansion/divide_and_conquer.py",
    "src/sciona/atoms/expansion/divide_and_conquer/references.json",
    "src/sciona/probes/expansion/divide_and_conquer.py",
    "tests/test_expansion_divide_and_conquer.py",
}


def test_divide_and_conquer_expansion_review_bundle_matches_runtime_surface() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))

    declaration_fqdns = {fqdn for fqdn, _, _ in DIVIDE_AND_CONQUER_DECLARATIONS.values()}
    probe_fqdns = {target.atom_fqdn for target in DIVIDE_AND_CONQUER_PROBE_TARGETS}

    assert bundle["provider_repo"] == "sciona-atoms"
    assert bundle["family_batch"] == "divide_and_conquer_expansion"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass"
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] == "reviewed_with_limits"
    assert declaration_fqdns == probe_fqdns
    assert {row["atom_key"] for row in bundle["rows"]} == declaration_fqdns
    assert not any(row["atom_key"].startswith("divide_and_conquer/") for row in bundle["rows"])

    authoritative_sources = {(source["kind"], source["path"]) for source in bundle["authoritative_sources"]}
    assert authoritative_sources == EXPECTED_AUTHORITATIVE_SOURCES

    for _, rel in EXPECTED_AUTHORITATIVE_SOURCES:
        assert (ROOT / rel).exists()

    for row in bundle["rows"]:
        assert row["review_status"] == "reviewed"
        assert row["review_semantic_verdict"] == "pass"
        assert row["review_developer_semantic_verdict"] == "pass_with_limits"
        assert row["trust_readiness"] == "catalog_ready"
        assert row["review_record_path"] == "data/review_bundles/divide_and_conquer_expansion.review_bundle.json"
        assert set(row["source_paths"]) == EXPECTED_ROW_SOURCE_PATHS
        for rel in row["source_paths"]:
            assert (ROOT / rel).exists()
