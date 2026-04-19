from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "numpy_random_search_pubrev_024_059.review_bundle.json"

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


def test_numpy_random_search_review_bundle_covers_expected_atoms() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text())

    assert bundle["provider_repo"] == "sciona-atoms"
    assert bundle["family_batch"] == "numpy_random_search_pubrev_024_059"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] in {"pass", "pass_with_limits"}
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] in {"catalog_ready", "reviewed_with_limits"}
    assert bundle["review_record_path"] == "data/review_bundles/numpy_random_search_pubrev_024_059.review_bundle.json"

    rows = bundle["rows"]
    assert {row["atom_key"] for row in rows} == EXPECTED_ATOMS

    for row in rows:
        assert row["atom_name"] == row["atom_key"]
        assert row["review_status"] == "reviewed"
        assert row["review_semantic_verdict"] in {"pass", "pass_with_limits"}
        assert row["review_developer_semantic_verdict"] == "pass_with_limits"
        assert row["trust_readiness"] == "catalog_ready"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["parity_test_status"] == "pass"
        assert isinstance(row["risk_score"], int)
        assert isinstance(row["acceptability_score"], int)
        for rel_path in row["source_paths"]:
            assert (ROOT / rel_path).exists()
