from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "data" / "review_bundles"
MANIFEST = json.loads((ROOT / "data" / "audit_manifest.json").read_text())
MANIFEST_KEYS = {row["atom_key"] for row in MANIFEST["atoms"]}


def _load_bundle(path: Path) -> dict:
    data = json.loads(path.read_text())
    assert data["review_record_path"] == f"data/review_bundles/{path.name}"
    return data


def test_review_bundles_have_concrete_review_metadata() -> None:
    bundle_paths = sorted(BUNDLE_DIR.glob("*.json"))
    assert bundle_paths

    for path in bundle_paths:
        bundle = _load_bundle(path)

        assert bundle["provider_repo"] == "sciona-atoms"
        assert bundle["review_status"] == "reviewed"
        assert bundle["review_semantic_verdict"] in {"pass", "pass_with_limits"}
        assert bundle["review_developer_semantic_verdict"] in {"pass", "pass_with_limits"}
        assert bundle["trust_readiness"] in {"catalog_ready", "reviewed_with_limits"}
        assert bundle["authoritative_sources"]
        assert bundle["rows"]

        for source in bundle["authoritative_sources"]:
            assert source["kind"]
            assert source.get("path") or source.get("repo")

        for row in bundle["rows"]:
            assert row["atom_key"] in MANIFEST_KEYS
            assert row["review_status"] == "reviewed"
            assert row["review_semantic_verdict"] in {"pass", "pass_with_limits"}
            assert row["review_developer_semantic_verdict"] in {"pass", "pass_with_limits"}
            assert row["trust_readiness"] in {"catalog_ready", "needs_followup"}
            assert row["review_record_path"] == bundle["review_record_path"]
            assert row["source_paths"]
            for rel in row["source_paths"]:
                assert (ROOT / rel).exists()
            manifest_row = next(r for r in MANIFEST["atoms"] if r["atom_key"] == row["atom_key"])
            assert manifest_row["review_status"] in {"approved", "reviewed_pending", "missing"}


def test_scipy_review_bundles_cover_expected_rows() -> None:
    expected = {
        "scipy_fft.review_bundle.json": {"scipy/fft:dct", "scipy/fft:idct"},
        "scipy_integrate.review_bundle.json": {
            "scipy/integrate:quad",
            "scipy/integrate:simpson",
            "scipy/integrate:solve_ivp",
        },
        "scipy_interpolate.review_bundle.json": {
            "sciona.atoms.scipy.interpolate.cubicsplinefit",
            "sciona.atoms.scipy.interpolate.rbfinterpolatorfit",
        },
        "scipy_linalg.review_bundle.json": {
            "scipy/linalg:det",
            "scipy/linalg:inv",
            "scipy/linalg:lu_factor",
            "scipy/linalg:lu_solve",
            "scipy/linalg:solve",
        },
        "scipy_optimize.review_bundle.json": {
            "scipy/optimize:curve_fit",
            "scipy/optimize:linprog",
            "scipy/optimize:minimize",
            "scipy/optimize:root",
        },
        "scipy_signal.review_bundle.json": {
            "scipy/signal:butter",
            "scipy/signal:cheby1",
            "scipy/signal:cheby2",
            "scipy/signal:firwin",
            "scipy/signal:freqz",
            "scipy/signal:lfilter",
            "scipy/signal:sosfilt",
        },
        "scipy_spatial.review_bundle.json": {
            "sciona.atoms.scipy.spatial.delaunay_triangulation",
            "sciona.atoms.scipy.spatial.voronoi_tessellation",
        },
        "scipy_stats.review_bundle.json": {
            "scipy/stats:describe",
            "scipy/stats:pearsonr",
            "scipy/stats:spearmanr",
            "scipy/stats:ttest_ind",
        },
    }

    for filename, atom_keys in expected.items():
        bundle = _load_bundle(BUNDLE_DIR / filename)
        assert {row["atom_key"] for row in bundle["rows"]} == atom_keys
