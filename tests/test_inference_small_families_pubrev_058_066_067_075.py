from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sciona.atoms.audit_review_bundles import load_review_bundle_entries, merge_audit_manifest_with_review_bundles
from sciona.atoms.inference.advancedvi import (
    evaluate_log_probability_density,
    gradient_oracle_evaluation,
    optimizationlooporchestration,
)
from sciona.atoms.inference.bayes_rs import bernoulli_probabilistic_oracle
from sciona.atoms.inference.conjugate_priors.beta_binom import posterior_randmodel, posterior_randmodel_weighted
from sciona.atoms.inference.jax_advi.optimize_advi import meanfieldvariationalfit, posteriordrawsampling


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = (
    ROOT
    / "data"
    / "review_bundles"
    / "inference_small_families_pubrev_058_066_067_075.review_bundle.json"
)
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"
MANIFEST_PATH = ROOT / "data" / "audit_manifest.json"

TARGET_ATOMS = {
    "sciona.atoms.inference.advancedvi.core.evaluate_log_probability_density",
    "sciona.atoms.inference.advancedvi.core.gradient_oracle_evaluation",
    "sciona.atoms.inference.advancedvi.core.optimizationlooporchestration",
    "sciona.atoms.inference.conjugate_priors.beta_binom.posterior_randmodel",
    "sciona.atoms.inference.conjugate_priors.beta_binom.posterior_randmodel_weighted",
    "sciona.atoms.inference.jax_advi.optimize_advi.meanfieldvariationalfit",
    "sciona.atoms.inference.jax_advi.optimize_advi.posteriordrawsampling",
    "sciona.atoms.inference.bayes_rs.bernoulli.bernoulli_probabilistic_oracle",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pubrev_inference_advancedvi_atoms_have_runtime_semantics() -> None:
    q = np.array([1.0, -1.0, np.log(0.5), np.log(2.0)])
    z = np.array([1.5, 1.0])
    expected = -np.log(2.0 * np.pi) - np.log(0.5) - np.log(2.0) - 0.5 * (1.0**2 + 1.0**2)

    assert np.isclose(evaluate_log_probability_density(q, z), expected)
    with pytest.raises(ValueError, match="equal non-empty halves"):
        evaluate_log_probability_density(np.array([0.0, 1.0, 2.0]), np.array([0.0]))

    grad, value, state_out, rng_out = gradient_oracle_evaluation(
        np.array([7]),
        lambda x: float(np.sum((x - 2.0) ** 2)),
        "finite_difference",
        np.zeros((2, 2)),
        np.array([3.0]),
        np.array([[2.0, 4.0], [1.0, -1.0]]),
        lambda g: g.reshape(4),
    )
    assert np.allclose(grad, np.array([0.0, 4.0, -2.0, -6.0]), atol=1e-5)
    assert np.isclose(value, 14.0)
    assert np.array_equal(state_out, np.array([3.0]))
    assert np.array_equal(rng_out, np.array([7]))

    optimized, rng_state, final_q = optimizationlooporchestration(
        None,
        20,
        lambda q_value: float(np.sum((q_value - 1.5) ** 2)),
        np.array([0.0, 3.0]),
        np.array([11]),
    )
    assert np.allclose(optimized, np.array([1.5, 1.5]), atol=1e-4)
    assert np.array_equal(final_q, optimized)
    assert np.array_equal(rng_state, np.array([11]))


def test_pubrev_inference_beta_binom_and_bernoulli_atoms_are_checked() -> None:
    prior = np.array([2.0, 3.0])
    data = np.array([1.0, 0.0, 1.0, 1.0])

    assert np.allclose(posterior_randmodel(prior, np.empty((0,)), data), np.array([5.0, 4.0]))
    assert np.allclose(
        posterior_randmodel_weighted(prior, np.empty((0,)), data, np.array([1.0, 0.5, 2.0, 0.0])),
        np.array([5.0, 3.5]),
    )
    with pytest.raises(ValueError, match="binary observations"):
        posterior_randmodel(prior, np.empty((0,)), np.array([0.5]))
    with pytest.raises(ValueError, match="one weight per observation"):
        posterior_randmodel_weighted(prior, np.empty((0,)), data, np.array([1.0]))

    log_lik = bernoulli_probabilistic_oracle(0.25, np.array([0.0, 1.0]))
    assert np.allclose(log_lik, np.log(np.array([0.75, 0.25])))
    with pytest.raises(ValueError, match="Bernoulli observations"):
        bernoulli_probabilistic_oracle(0.25, np.array([2.0]))


def test_pubrev_inference_jax_advi_atoms_run_without_external_jax_advi_dependency() -> None:
    def log_prior(params: dict[str, np.ndarray]) -> float:
        theta = params["theta"]
        return float(-0.5 * np.sum(theta**2))

    def log_lik(params: dict[str, np.ndarray]) -> float:
        theta = params["theta"]
        return float(-20.0 * np.sum((theta - 1.0) ** 2))

    free_means, free_sds, objective_fun, rng_state = meanfieldvariationalfit(
        {"theta": (1,)},
        log_prior,
        log_lik,
        M=12,
        seed=5,
        var_param_inits={"theta": (np.array([0.0]), np.array([-1.0]))},
    )

    assert free_means["theta"].shape == (1,)
    assert free_means["theta"][0] > 0.5
    assert np.all(free_sds["theta"] > 0.0)
    assert np.isfinite(objective_fun())
    assert rng_state == 6

    draws_a, next_rng = posteriordrawsampling(free_means, free_sds, {}, 4, None, 17)
    draws_b, _ = posteriordrawsampling(free_means, free_sds, {}, 4, None, 17)
    assert next_rng == 18
    assert draws_a["theta"].shape == (4, 1)
    assert np.allclose(draws_a["theta"], draws_b["theta"])

    transformed, _ = posteriordrawsampling(
        free_means,
        free_sds,
        {},
        3,
        lambda sample: float(sample["theta"][0] ** 2),
        21,
    )
    assert transformed.shape == (3,)
    assert np.all(transformed >= 0.0)


def test_pubrev_inference_references_are_canonical_and_registry_backed() -> None:
    reference_paths = [
        ROOT / "src" / "sciona" / "atoms" / "inference" / "advancedvi" / "references.json",
        ROOT / "src" / "sciona" / "atoms" / "inference" / "conjugate_priors" / "beta_binom" / "references.json",
        ROOT / "src" / "sciona" / "atoms" / "inference" / "jax_advi" / "optimize_advi" / "references.json",
        ROOT / "src" / "sciona" / "atoms" / "inference" / "bayes_rs" / "references.json",
    ]
    registry_ids = set(_json(REGISTRY_PATH)["references"])
    observed_atoms: set[str] = set()

    for path in reference_paths:
        payload = _json(path)
        observed_atoms.update(payload["atoms"])
        for atom_key, metadata in payload["atoms"].items():
            assert atom_key.startswith("sciona.atoms.inference.")
            assert "@sciona/atoms/" not in atom_key
            assert metadata["references"]
            assert {ref["ref_id"] for ref in metadata["references"]}.issubset(registry_ids)

    assert TARGET_ATOMS <= observed_atoms


def test_pubrev_inference_review_bundle_is_publishable_and_mergeable() -> None:
    bundle = _json(BUNDLE_PATH)

    assert bundle["provider_repo"] == "sciona-atoms"
    assert bundle["family_batch"] == "inference_small_families_pubrev_058_066_067_075"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass"
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] == "catalog_ready"
    assert bundle["blocking_findings"] == []
    assert bundle["required_actions"] == []
    assert {row["atom_key"] for row in bundle["rows"]} == TARGET_ATOMS

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
    assert summary["bundle_entry_count"] == len(TARGET_ATOMS)
    assert summary["skipped_unresolved_atom_count"] == 0
