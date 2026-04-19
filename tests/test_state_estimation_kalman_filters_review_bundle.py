from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "state_estimation.review_bundle.json"

EXPECTED_TRACKER_KEY = "sciona.atoms.state_estimation.kalman_filters.track_linear_gaussian_state"
EXPECTED_TRACKER_SOURCE_PATHS = {
    "src/sciona/atoms/state_estimation/kalman_filters/atoms.py",
    "src/sciona/atoms/state_estimation/kalman_filters/references.json",
    "tests/test_tracking_contract_wrappers.py",
    "tests/test_kalman_filter_contract_references_metadata.py",
}


def test_state_estimation_bundle_covers_kalman_tracker_contract_atom() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    rows = [row for row in bundle["rows"] if row["atom_key"] == EXPECTED_TRACKER_KEY]

    assert len(rows) == 1
    row = rows[0]
    assert row["review_status"] == "reviewed"
    assert row["review_semantic_verdict"] == "pass"
    assert row["review_developer_semantic_verdict"] == "pass_with_limits"
    assert row["trust_readiness"] == "catalog_ready"
    assert set(row["source_paths"]) == EXPECTED_TRACKER_SOURCE_PATHS
    assert row["review_record_path"] == "data/review_bundles/state_estimation.review_bundle.json"

    for rel in EXPECTED_TRACKER_SOURCE_PATHS:
        assert (ROOT / rel).exists()
