from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "belief_propagation.review_bundle.json"


def test_belief_propagation_review_bundle_uses_current_inference_fqdns() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))

    assert bundle["provider_repo"] == "sciona-atoms"
    assert bundle["family_batch"] == "belief_propagation"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass"
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] == "catalog_ready"

    assert {row["atom_key"] for row in bundle["rows"]} == {
        "sciona.atoms.inference.belief_propagation.loopy_bp.initialize_message_passing_state",
        "sciona.atoms.inference.belief_propagation.loopy_bp.run_loopy_message_passing_and_belief_query",
    }
    assert not any(row["atom_key"].startswith("belief_propagation/") for row in bundle["rows"])

    for row in bundle["rows"]:
        assert row["review_record_path"] == "data/review_bundles/belief_propagation.review_bundle.json"
        for rel in row["source_paths"]:
            assert (ROOT / rel).exists()
