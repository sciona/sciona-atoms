from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "numpy_polynomial.review_bundle.json"

EXPECTED_ATOM_KEYS = {
    "sciona.atoms.numpy.polynomial.polyadd",
    "sciona.atoms.numpy.polynomial.polyder",
    "sciona.atoms.numpy.polynomial.polyfit",
    "sciona.atoms.numpy.polynomial.polyint",
    "sciona.atoms.numpy.polynomial.polymul",
    "sciona.atoms.numpy.polynomial.polyroots",
    "sciona.atoms.numpy.polynomial.polyval",
}

EXPECTED_SOURCE_PATHS = {
    "src/sciona/atoms/numpy/polynomial.py",
    "src/sciona/probes/numpy/polynomial.py",
    "tests/test_numpy_polynomial_import_smoke.py",
}


def test_numpy_polynomial_review_bundle_covers_current_runtime_fqdns() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))

    assert bundle["provider_repo"] == "sciona-atoms"
    assert bundle["family_batch"] == "numpy_polynomial"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass"
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] == "catalog_ready"

    rows = bundle["rows"]
    assert {row["atom_key"] for row in rows} == EXPECTED_ATOM_KEYS
    assert not any(row["atom_key"].startswith("numpy/polynomial:") for row in rows)

    for row in rows:
        assert row["review_status"] == "reviewed"
        assert row["review_semantic_verdict"] == "pass"
        assert row["review_developer_semantic_verdict"] == "pass_with_limits"
        assert row["trust_readiness"] == "catalog_ready"
        assert row["review_record_path"] == "data/review_bundles/numpy_polynomial.review_bundle.json"
        assert set(row["source_paths"]) == EXPECTED_SOURCE_PATHS
        for rel in row["source_paths"]:
            assert (ROOT / rel).exists()
