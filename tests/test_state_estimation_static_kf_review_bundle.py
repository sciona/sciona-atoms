from __future__ import annotations

import json
from pathlib import Path


DRAFT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = DRAFT_ROOT / "data" / "review_bundles" / "state_estimation.review_bundle.json"
STATIC_KF_ROOT = (
    DRAFT_ROOT
    / "src"
    / "sciona"
    / "atoms"
    / "state_estimation"
    / "kalman_filters"
    / "static_kf"
)

EXPECTED_STATIC_KF_KEYS = {
    "sciona.atoms.state_estimation.kalman_filters.static_kf.exposecovariance",
    "sciona.atoms.state_estimation.kalman_filters.static_kf.exposelatentmean",
    "sciona.atoms.state_estimation.kalman_filters.static_kf.initializelineargaussianstatemodel",
    "sciona.atoms.state_estimation.kalman_filters.static_kf.predictlatentstate",
    "sciona.atoms.state_estimation.kalman_filters.static_kf.updatewithmeasurement",
}

EXPECTED_STATIC_KF_SOURCE_PATHS = {
    "src/sciona/atoms/state_estimation/kalman_filters/static_kf/atoms.py",
    "src/sciona/atoms/state_estimation/kalman_filters/static_kf/references.json",
    "src/sciona/atoms/state_estimation/kalman_filters/static_kf/cdg.json",
    "src/sciona/atoms/state_estimation/kalman_filters/static_kf/matches.json",
    "tests/test_static_kf_references_metadata.py",
}


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def test_state_estimation_static_kf_review_bundle_covers_all_current_runtime_fqdns() -> None:
    bundle = _load_json(BUNDLE_PATH)
    references = _load_json(STATIC_KF_ROOT / "references.json")
    cdg = _load_json(STATIC_KF_ROOT / "cdg.json")
    matches = _load_json(STATIC_KF_ROOT / "matches.json")

    static_kf_rows = [
        row
        for row in bundle["rows"]
        if "state_estimation.kalman_filters.static_kf." in row["atom_key"]
    ]

    assert {row["atom_key"] for row in static_kf_rows} == EXPECTED_STATIC_KF_KEYS
    assert not any(row["atom_key"].startswith("kalman_filters/static_kf:") for row in bundle["rows"])

    reference_fqdns = {key.partition("@")[0] for key in references["atoms"]}
    assert reference_fqdns == EXPECTED_STATIC_KF_KEYS

    expected_leaves = {key.rsplit(".", 1)[-1] for key in EXPECTED_STATIC_KF_KEYS}
    assert set(cdg["nodes"][0]["children"]) == expected_leaves

    matched_predicates = {entry["pdg_node"]["predicate_id"] for entry in matches}
    assert matched_predicates
    assert matched_predicates <= expected_leaves
    assert "initializelineargaussianstatemodel" in matched_predicates

    for row in static_kf_rows:
        assert set(row["source_paths"]) == EXPECTED_STATIC_KF_SOURCE_PATHS
        assert row["review_status"] == "reviewed"
        assert row["review_semantic_verdict"] == "pass"
        assert row["review_developer_semantic_verdict"] == "pass_with_limits"
        assert row["trust_readiness"] == "catalog_ready"
        assert row["review_record_path"] == "data/review_bundles/state_estimation.review_bundle.json"
