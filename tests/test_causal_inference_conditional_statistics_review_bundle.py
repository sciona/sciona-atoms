from __future__ import annotations

import json
from pathlib import Path

from sciona.atoms.audit_review_bundles import (
    VALID_ACCEPTABILITY_BANDS,
    VALID_PARITY_COVERAGE_LEVELS,
)

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "causal_inference_conditional_statistics.review_bundle.json"
CDG_PATH = ROOT / "src" / "sciona" / "atoms" / "causal_inference" / "conditional_statistics" / "cdg.json"
MANIFEST_PATH = ROOT / "data" / "audit_manifest.json"

EXPECTED_ATOM_NAMES = {
    "sciona.atoms.causal_inference.conditional_statistics.conditional_noise_entropy_variance",
    "sciona.atoms.causal_inference.conditional_statistics.conditional_noise_skewness_variance",
    "sciona.atoms.causal_inference.conditional_statistics.conditional_noise_kurtosis_variance",
    "sciona.atoms.causal_inference.conditional_statistics.conditional_distribution_similarity",
}


def test_bundle_exists_and_has_four_atoms() -> None:
    assert BUNDLE_PATH.exists()
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    assert len(bundle["rows"]) == 4
    assert {row["atom_key"] for row in bundle["rows"]} == EXPECTED_ATOM_NAMES


def test_bundle_level_fields() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    assert bundle["provider_repo"] == "sciona-atoms"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass"
    assert bundle["trust_readiness"] == "reviewed_with_limits"


def test_each_row_has_correct_review_metadata() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    for row in bundle["rows"]:
        assert row["review_status"] == "reviewed"
        assert row["review_semantic_verdict"] == "pass"
        assert row["trust_readiness"] == "catalog_ready"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"


def test_source_paths_exist() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    for row in bundle["rows"]:
        for rel in row["source_paths"]:
            assert (ROOT / rel).exists(), f"missing: {rel}"


def test_scores_are_db_compatible_integers() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    for row in bundle["rows"]:
        assert isinstance(row["risk_score"], int), f"risk_score not int: {row['risk_score']}"
        assert isinstance(row["acceptability_score"], int), f"acceptability_score not int: {row['acceptability_score']}"


def test_acceptability_band_in_db_taxonomy() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    for row in bundle["rows"]:
        assert row["acceptability_band"] in VALID_ACCEPTABILITY_BANDS


def test_parity_coverage_level_in_db_enum() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    for row in bundle["rows"]:
        assert row["parity_coverage_level"] in VALID_PARITY_COVERAGE_LEVELS


def test_cdg_node_names_are_fqdn_derivable() -> None:
    cdg = json.loads(CDG_PATH.read_text(encoding="utf-8"))
    for node in cdg["nodes"]:
        if node.get("status") == "atomic":
            name = node["name"]
            assert " " not in name, f"CDG node name contains spaces: '{name}'"
            assert name == name.lower(), f"CDG node name not lowercase: '{name}'"
