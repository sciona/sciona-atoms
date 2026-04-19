from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "numpy_fft_pubrev_015.review_bundle.json"

EXPECTED_ATOM_KEYS = {
    "sciona.atoms.numpy.fft.fft",
    "sciona.atoms.numpy.fft.fftfreq",
    "sciona.atoms.numpy.fft.fftn",
    "sciona.atoms.numpy.fft.fftshift",
    "sciona.atoms.numpy.fft.hfft",
    "sciona.atoms.numpy.fft.ifft",
    "sciona.atoms.numpy.fft.ifftn",
    "sciona.atoms.numpy.fft.irfft",
    "sciona.atoms.numpy.fft.rfft",
}

EXPECTED_SOURCE_PATHS = {
    "src/sciona/atoms/numpy/fft.py",
    "src/sciona/probes/numpy/fft.py",
    "src/sciona/atoms/numpy/references.json",
    "tests/test_numpy_fft_behavior.py",
    "tests/test_numpy_fft_review_bundle.py",
}


def test_numpy_fft_pubrev_015_review_bundle_is_publishable() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))

    assert bundle["provider_repo"] == "sciona-atoms"
    assert bundle["family_batch"] == "numpy_fft_pubrev_015"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass"
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] == "catalog_ready"
    assert bundle["blocking_findings"] == []
    assert bundle["required_actions"] == []

    rows = bundle["rows"]
    assert {row["atom_key"] for row in rows} == EXPECTED_ATOM_KEYS

    for row in rows:
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
        assert row["review_record_path"] == "data/review_bundles/numpy_fft_pubrev_015.review_bundle.json"
        assert set(row["source_paths"]) == EXPECTED_SOURCE_PATHS
        for rel_path in row["source_paths"]:
            assert (ROOT / rel_path).exists()
