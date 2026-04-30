from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import norm

from sciona.atoms.audit_review_bundles import load_review_bundle_entries


ROOT = Path(__file__).resolve().parents[1]
FAMILY_DIR = ROOT / "src" / "sciona" / "atoms" / "inference" / "bayesian_bandits"
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "inference_bayesian_bandits.review_bundle.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_ATOMS = {
    "sciona.atoms.inference.bayesian_bandits.initialize_beta_beliefs",
    "sciona.atoms.inference.bayesian_bandits.beta_bernoulli_update",
    "sciona.atoms.inference.bayesian_bandits.thompson_sample_beta",
    "sciona.atoms.inference.bayesian_bandits.ucb_scores",
    "sciona.atoms.inference.bayesian_bandits.bayesian_ucb",
    "sciona.atoms.inference.bayesian_bandits.probability_of_improvement",
    "sciona.atoms.inference.bayesian_bandits.expected_improvement",
    "sciona.atoms.inference.bayesian_bandits.select_best_arm",
    "sciona.atoms.inference.bayesian_bandits.epsilon_greedy_select",
    "sciona.atoms.inference.bayesian_bandits.update_arm_statistics",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_beta_belief_initialization_and_update() -> None:
    from sciona.atoms.inference.bayesian_bandits import (
        beta_bernoulli_update,
        initialize_beta_beliefs,
    )

    alphas, betas = initialize_beta_beliefs(3, prior_alpha=2.0, prior_beta=4.0)
    assert np.allclose(alphas, np.array([2.0, 2.0, 2.0]))
    assert np.allclose(betas, np.array([4.0, 4.0, 4.0]))

    assert beta_bernoulli_update(2.0, 4.0, 1) == (3.0, 4.0)
    assert beta_bernoulli_update(2.0, 4.0, 0) == (2.0, 5.0)


def test_sampling_and_bandit_scores() -> None:
    from sciona.atoms.inference.bayesian_bandits import (
        bayesian_ucb,
        select_best_arm,
        thompson_sample_beta,
        ucb_scores,
    )

    alphas = np.array([2.0, 5.0], dtype=np.float64)
    betas = np.array([3.0, 2.0], dtype=np.float64)

    samples_a = thompson_sample_beta(alphas, betas, np.random.default_rng(42))
    samples_b = thompson_sample_beta(alphas, betas, np.random.default_rng(42))
    assert samples_a.shape == alphas.shape
    assert np.all((samples_a >= 0.0) & (samples_a <= 1.0))
    assert np.allclose(samples_a, samples_b)

    quantiles = bayesian_ucb(alphas, betas, 0.9)
    assert np.allclose(quantiles, beta_distribution.ppf(0.9, alphas, betas))

    means = np.array([0.8, 0.5, 0.1], dtype=np.float64)
    counts = np.array([4, 1, 0], dtype=np.int64)
    scores = ucb_scores(means, counts, total_count=5, c=2.0**0.5)
    assert np.isinf(scores[2])
    assert np.allclose(scores[:2], means[:2] + np.sqrt(2.0 * np.log(5.0) / counts[:2]))
    assert select_best_arm(scores) == 2


def test_gaussian_acquisition_functions_handle_zero_variance() -> None:
    from sciona.atoms.inference.bayesian_bandits import (
        expected_improvement,
        probability_of_improvement,
    )

    mean = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    std = np.array([1.0, 0.0, 2.0], dtype=np.float64)
    best = 0.5
    xi = 0.1

    pi = probability_of_improvement(mean, std, best, xi)
    expected_pi = np.array([
        norm.cdf((0.0 - best - xi) / 1.0),
        1.0 if 1.0 - best - xi > 0.0 else 0.0,
        norm.cdf((2.0 - best - xi) / 2.0),
    ])
    assert np.allclose(pi, expected_pi)

    ei = expected_improvement(mean, std, best, xi)
    z0 = (0.0 - best - xi) / 1.0
    z2 = (2.0 - best - xi) / 2.0
    expected_ei = np.array([
        (0.0 - best - xi) * norm.cdf(z0) + norm.pdf(z0),
        1.0 - best - xi,
        (2.0 - best - xi) * norm.cdf(z2) + 2.0 * norm.pdf(z2),
    ])
    assert np.allclose(ei, expected_ei)
    assert np.all(ei >= 0.0)


def test_action_selection_and_incremental_statistics() -> None:
    from sciona.atoms.inference.bayesian_bandits import (
        epsilon_greedy_select,
        select_best_arm,
        update_arm_statistics,
    )

    values = np.array([0.1, 0.4, 0.3], dtype=np.float64)
    assert select_best_arm(values) == 1
    assert epsilon_greedy_select(values, 0.0, np.random.default_rng(7)) == 1
    assert 0 <= epsilon_greedy_select(values, 1.0, np.random.default_rng(7)) < len(values)

    means = np.array([0.5, 0.0], dtype=np.float64)
    counts = np.array([2, 0], dtype=np.int64)
    updated_means, updated_counts = update_arm_statistics(means, counts, arm=0, reward=1.0)
    assert np.array_equal(updated_counts, np.array([3, 0]))
    assert np.allclose(updated_means, np.array([2.0 / 3.0, 0.0]))
    assert np.array_equal(counts, np.array([2, 0]))
    assert np.allclose(means, np.array([0.5, 0.0]))


def test_references_are_registry_backed_and_line_anchored() -> None:
    refs = _json(FAMILY_DIR / "references.json")
    registry_ids = set(_json(REGISTRY_PATH)["references"])
    source_lines = (FAMILY_DIR / "atoms.py").read_text(encoding="utf-8").splitlines()

    assert {key.partition("@")[0] for key in refs["atoms"]} == EXPECTED_ATOMS
    for key, entry in refs["atoms"].items():
        fqdn, _, anchor = key.partition("@")
        rel_path, _, line_text = anchor.partition(":")
        line_number = int(line_text)
        assert fqdn in EXPECTED_ATOMS
        assert rel_path == "sciona/atoms/inference/bayesian_bandits/atoms.py"
        assert f"def {fqdn.rsplit('.', 1)[-1]}" in source_lines[line_number - 1]
        assert entry["references"]
        for ref in entry["references"]:
            assert ref["ref_id"] in registry_ids
            assert ref["match_metadata"]["match_type"] == "manual"
            assert ref["match_metadata"]["notes"]


def test_cdg_and_review_bundle_cover_all_atoms() -> None:
    cdg = _json(FAMILY_DIR / "cdg.json")
    atomic_nodes = {node["name"]: node for node in cdg["nodes"] if node.get("status") == "atomic"}
    assert {f"sciona.atoms.inference.bayesian_bandits.{name}" for name in atomic_nodes} == EXPECTED_ATOMS
    for name, node in atomic_nodes.items():
        assert node["node_id"] == name
        assert node["inputs"]
        assert node["outputs"] == [{"name": "result", "type_desc": node["outputs"][0]["type_desc"]}]

    bundle = _json(BUNDLE_PATH)
    assert bundle["provider_repo"] == "sciona-atoms"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass"
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert bundle["blocking_findings"] == []
    assert bundle["required_actions"] == []
    assert {row["atom_key"] for row in bundle["rows"]} == EXPECTED_ATOMS

    for row in bundle["rows"]:
        assert row["atom_name"] == row["atom_key"]
        assert row["trust_readiness"] == "catalog_ready"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["parity_test_status"] == "pass"
        assert isinstance(row["risk_score"], int)
        assert isinstance(row["acceptability_score"], int)
        for rel_path in row["source_paths"]:
            assert (ROOT / rel_path).exists(), rel_path

    entries = load_review_bundle_entries(BUNDLE_PATH)
    assert [entry.atom_name for entry in entries] == sorted(EXPECTED_ATOMS)
