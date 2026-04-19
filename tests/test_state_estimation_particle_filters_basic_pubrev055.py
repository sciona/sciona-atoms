from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sciona.atoms.audit_review_bundles import (
    VALID_ACCEPTABILITY_BANDS,
    VALID_PARITY_COVERAGE_LEVELS,
    load_review_bundle_entries,
    merge_audit_manifest_with_review_bundles,
)
from sciona.atoms.state_estimation.particle_filters.basic import (
    hypothesis_propagation_kernel,
    likelihood_reweight_kernel,
    resample_and_hypothesis_distribution_projection,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = (
    ROOT
    / "data"
    / "review_bundles"
    / "state_estimation_particle_filters_basic_pubrev_055.review_bundle.json"
)
REFERENCES_PATH = (
    ROOT
    / "src"
    / "sciona"
    / "atoms"
    / "state_estimation"
    / "particle_filters"
    / "references.json"
)
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"
MANIFEST_PATH = ROOT / "data" / "audit_manifest.json"

TARGET_ATOMS = (
    "sciona.atoms.state_estimation.particle_filters.basic.filter_step_preparation_and_dispatch",
    "sciona.atoms.state_estimation.particle_filters.basic.hypothesis_propagation_kernel",
    "sciona.atoms.state_estimation.particle_filters.basic.likelihood_reweight_kernel",
    "sciona.atoms.state_estimation.particle_filters.basic.resample_and_hypothesis_distribution_projection",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pubrev055_propagation_uses_control_scale_and_normalized_weights() -> None:
    prior = {
        "particles": np.array([0.0, 1.0, 2.0]),
        "weights": np.array([2.0, 1.0, 1.0]),
        "rng_seed": 13,
    }

    near_deterministic, weights, rng_next = hypothesis_propagation_kernel(
        prior,
        {"process_scale": 1e-9},
        0.5,
        np.array([13], dtype=np.int64),
    )
    noisy, _, _ = hypothesis_propagation_kernel(
        prior,
        {"process_scale": 0.5},
        0.5,
        np.array([13], dtype=np.int64),
    )

    assert np.allclose(near_deterministic, prior["particles"] + 0.5, atol=1e-6)
    assert not np.allclose(noisy, near_deterministic)
    assert np.allclose(weights, np.array([0.5, 0.25, 0.25]))
    assert rng_next.tolist() == [14]


def test_pubrev055_likelihood_uses_observation_scale_and_normalizes() -> None:
    particles = np.array([-1.0, 0.0, 1.0])
    weights = np.ones(3) / 3.0

    tight, tight_log_likelihood = likelihood_reweight_kernel(
        particles,
        weights,
        np.array([0.0]),
        {"observation_scale": 0.1},
    )
    broad, broad_log_likelihood = likelihood_reweight_kernel(
        particles,
        weights,
        np.array([0.0]),
        {"observation_scale": 10.0},
    )

    assert np.isclose(tight.sum(), 1.0)
    assert np.isclose(broad.sum(), 1.0)
    assert tight[1] > broad[1]
    assert np.isfinite(tight_log_likelihood)
    assert np.isfinite(broad_log_likelihood)


def test_pubrev055_resampling_returns_equal_weight_posterior_and_ess() -> None:
    posterior, trace = resample_and_hypothesis_distribution_projection(
        np.array([10.0, 20.0]),
        np.array([1.0, 3.0]),
        np.array([5], dtype=np.int64),
        -1.25,
    )

    assert posterior["particles"].shape == (2,)
    assert np.array_equal(posterior["weights"], np.array([0.5, 0.5]))
    assert posterior["rng_seed"] == 6
    assert trace["log_likelihood"] == -1.25
    assert np.isclose(trace["ess"], 1.0 / (0.25**2 + 0.75**2))


def test_pubrev055_invalid_weights_are_rejected() -> None:
    prior = {"particles": np.array([0.0, 1.0]), "weights": np.array([1.0]), "rng_seed": 3}

    with pytest.raises(ValueError, match="one entry per particle"):
        hypothesis_propagation_kernel(prior, {"process_scale": 0.1}, 0.0, np.array([3]))

    with pytest.raises(ValueError, match="finite and non-negative"):
        likelihood_reweight_kernel(
            np.array([0.0, 1.0]),
            np.array([2.0, -1.0]),
            np.array([0.0]),
            {"observation_scale": 1.0},
        )

    with pytest.raises(ValueError, match="disagree"):
        resample_and_hypothesis_distribution_projection(
            np.array([0.0, 1.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([4]),
            -0.5,
        )


def test_pubrev055_references_use_repo_local_registry_entries() -> None:
    payload = _json(REFERENCES_PATH)
    registry = _json(REGISTRY_PATH)["references"]
    registry_ids = set(registry)

    for atom in TARGET_ATOMS:
        refs = payload["atoms"][atom]["references"]
        assert [ref["ref_id"] for ref in refs] == [
            "gordon1993particle",
            "repo_particlefilters_jl",
        ]
        assert {ref["ref_id"] for ref in refs}.issubset(registry_ids)
        for ref in refs:
            metadata = ref["match_metadata"]
            assert metadata["match_type"] == "manual"
            assert metadata["confidence"] in {"high", "medium"}
            assert metadata["notes"]


def test_pubrev055_review_bundle_is_publishable_and_mergeable() -> None:
    bundle = _json(BUNDLE_PATH)

    assert bundle["provider_repo"] == "sciona-atoms"
    assert bundle["family_batch"] == "state_estimation_particle_filters_basic_pubrev_055"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass"
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] == "catalog_ready"
    assert bundle["blocking_findings"] == []
    assert bundle["required_actions"] == []
    assert {row["atom_key"] for row in bundle["rows"]} == set(TARGET_ATOMS)

    for row in bundle["rows"]:
        assert row["atom_name"] == row["atom_key"]
        assert row["review_status"] == "reviewed"
        assert row["trust_readiness"] == "catalog_ready"
        assert row["overall_verdict"] == "acceptable_with_limits"
        assert row["structural_status"] == "pass"
        assert row["semantic_status"] == "pass"
        assert row["runtime_status"] == "pass"
        assert row["developer_semantics_status"] == "pass_with_limits"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert isinstance(row["risk_score"], int)
        assert isinstance(row["acceptability_score"], int)
        assert row["acceptability_band"] in VALID_ACCEPTABILITY_BANDS
        assert row["parity_coverage_level"] in VALID_PARITY_COVERAGE_LEVELS
        assert row["parity_test_status"] == "pass"
        assert row["blocking_findings"] == []
        assert row["required_actions"] == []
        for rel_path in row["source_paths"]:
            assert (ROOT / rel_path).exists(), rel_path

    entries = load_review_bundle_entries(BUNDLE_PATH)
    assert [entry.atom_name for entry in entries] == sorted(TARGET_ATOMS)

    summary = merge_audit_manifest_with_review_bundles(
        manifest_path=MANIFEST_PATH,
        review_bundle_paths=[BUNDLE_PATH],
        dry_run=True,
    )
    assert summary["bundle_entry_count"] == 4
    assert summary["skipped_unresolved_atom_count"] == 0
