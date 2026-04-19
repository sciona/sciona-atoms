from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "causal_inference_feature_primitives.review_bundle.json"
CDG_PATH = ROOT / "src" / "sciona" / "atoms" / "causal_inference" / "feature_primitives" / "cdg.json"
MANIFEST_PATH = ROOT / "data" / "audit_manifest.json"

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

# DB-compatible acceptability bands (from existing manifest taxonomy)
VALID_ACCEPTABILITY_BANDS = {
    "acceptable_with_limits_candidate",
    "review_ready",
    "limited_acceptability",
    "broken_candidate",
    "misleading_candidate",
}

# DB enum for parity_coverage_level
VALID_PARITY_LEVELS = {
    "unknown",
    "none",
    "not_applicable",
    "positive_path",
    "positive_and_negative",
    "parity_or_usage_equivalent",
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


def test_scores_are_db_compatible_integers() -> None:
    """risk_score and acceptability_score must be integers (DB schema is INTEGER)."""
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    for row in bundle["rows"]:
        rs = row["risk_score"]
        assert isinstance(rs, int), f"risk_score must be int, got {type(rs).__name__}: {rs}"
        assert 0 <= rs <= 100, f"risk_score out of range: {rs}"

        acc = row["acceptability_score"]
        assert isinstance(acc, int), f"acceptability_score must be int, got {type(acc).__name__}: {acc}"
        assert 0 <= acc <= 100, f"acceptability_score out of range: {acc}"


def test_acceptability_band_in_db_taxonomy() -> None:
    """acceptability_band must use a value the backfill recognizes."""
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    for row in bundle["rows"]:
        band = row["acceptability_band"]
        assert band in VALID_ACCEPTABILITY_BANDS, (
            f"acceptability_band '{band}' not in DB taxonomy: {VALID_ACCEPTABILITY_BANDS}"
        )


def test_parity_coverage_level_in_db_enum() -> None:
    """parity_coverage_level must use a DB-recognized enum value."""
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    for row in bundle["rows"]:
        level = row["parity_coverage_level"]
        assert level in VALID_PARITY_LEVELS, (
            f"parity_coverage_level '{level}' not in DB enum: {VALID_PARITY_LEVELS}"
        )


def test_cdg_node_names_are_fqdn_derivable() -> None:
    """CDG atomic node 'name' must be the snake_case function name so IO backfill
    can derive a valid FQDN (not human-readable titles)."""
    cdg = json.loads(CDG_PATH.read_text(encoding="utf-8"))
    for node in cdg["nodes"]:
        if node.get("status") == "atomic":
            name = node["name"]
            assert " " not in name, f"CDG node name contains spaces: '{name}'"
            assert name == name.lower(), f"CDG node name not lowercase: '{name}'"
            assert name == node["node_id"], (
                f"CDG node name '{name}' != node_id '{node['node_id']}'"
            )


def test_manifest_scores_are_integers() -> None:
    """After manifest merge, scores must still be integers."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for atom in manifest.get("atoms", []):
        if "causal_inference" not in atom.get("atom_name", ""):
            continue
        if "risk_score" in atom:
            assert isinstance(atom["risk_score"], int), (
                f"manifest risk_score for {atom['atom_name']} is {type(atom['risk_score']).__name__}"
            )
        if "acceptability_score" in atom:
            assert isinstance(atom["acceptability_score"], int), (
                f"manifest acceptability_score for {atom['atom_name']} is {type(atom['acceptability_score']).__name__}"
            )
        if "acceptability_band" in atom:
            assert atom["acceptability_band"] in VALID_ACCEPTABILITY_BANDS, (
                f"manifest acceptability_band '{atom['acceptability_band']}' not in taxonomy"
            )
