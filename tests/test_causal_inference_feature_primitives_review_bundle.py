from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "causal_inference_feature_primitives.review_bundle.json"

EXPECTED_ATOM_NAMES = {
    "sciona.atoms.causal_inference.feature_primitives.igci_asymmetry_score",
    "sciona.atoms.causal_inference.feature_primitives.hsic_independence_test",
    "sciona.atoms.causal_inference.feature_primitives.knn_entropy_estimator",
    "sciona.atoms.causal_inference.feature_primitives.uniform_divergence",
    "sciona.atoms.causal_inference.feature_primitives.normalized_error_probability",
    "sciona.atoms.causal_inference.feature_primitives.discretize_and_bin",
    "sciona.atoms.causal_inference.feature_primitives.polyfit_nonlinearity_asymmetry",
    "sciona.atoms.causal_inference.feature_primitives.polyfit_residual_error",
}


def test_bundle_exists_and_has_eight_atoms() -> None:
    assert BUNDLE_PATH.exists()
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    assert len(bundle["rows"]) == 8
    assert {row["atom_key"] for row in bundle["rows"]} == EXPECTED_ATOM_NAMES


def test_bundle_level_fields() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    assert bundle["provider_repo"] == "sciona-atoms"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass"
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] == "reviewed_with_limits"
    assert bundle["review_record_path"] == "data/review_bundles/causal_inference_feature_primitives.review_bundle.json"


def test_each_row_has_correct_review_metadata() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    for row in bundle["rows"]:
        assert row["review_status"] == "reviewed"
        assert row["review_semantic_verdict"] == "pass"
        assert row["review_developer_semantic_verdict"] == "pass_with_limits"
        assert row["trust_readiness"] == "catalog_ready"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["review_record_path"] == bundle["review_record_path"]


def test_source_paths_exist() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    for row in bundle["rows"]:
        assert row["source_paths"]
        for rel in row["source_paths"]:
            assert (ROOT / rel).exists(), f"missing: {rel}"
