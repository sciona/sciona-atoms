from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np

from sciona.atoms.audit_review_bundles import merge_audit_manifest_with_review_bundles


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "mcmc_foundational.review_bundle.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"
REFERENCE_PATHS = [
    ROOT / "src" / "sciona" / "atoms" / "inference" / "mcmc_foundational" / "mini_mcmc" / "hmc" / "references.json",
    ROOT / "src" / "sciona" / "atoms" / "inference" / "mcmc_foundational" / "mini_mcmc" / "hmc_llm" / "references.json",
    ROOT / "src" / "sciona" / "atoms" / "inference" / "mcmc_foundational" / "mini_mcmc" / "nuts_llm" / "references.json",
    ROOT / "src" / "sciona" / "atoms" / "inference" / "mcmc_foundational" / "mini_mcmc" / "references.json",
]

SAFE_ATOMS = {
    "sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc.initializehmcstate",
    "sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc.leapfrogproposalkernel",
    "sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc_llm.hamiltoniantransitionkernel",
    "sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc_llm.initializehmckernelstate",
    "sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc_llm.initializesamplerrng",
    "sciona.atoms.inference.mcmc_foundational.mini_mcmc.nuts_llm.initializenutsstate",
}

HELD_ATOMS = {
    "sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc.metropolishmctransition",
    "sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc.runsamplingloop",
    "sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc_llm.collectposteriorchain",
    "sciona.atoms.inference.mcmc_foundational.mini_mcmc.nuts_llm.runnutstransitions",
    "sciona.atoms.inference.mcmc_foundational.mini_mcmc.nuts.nuts_recursive_tree_build",
}


def _standard_normal_logp(x: np.ndarray) -> float:
    values = np.asarray(x, dtype=np.float64)
    return float(-0.5 * np.dot(values, values))


def _import_leaf(fqdn: str):
    module_name, _, symbol_name = fqdn.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def test_pubrev008_safe_and_held_fqdns_are_importable() -> None:
    for fqdn in sorted(SAFE_ATOMS | HELD_ATOMS):
        assert callable(_import_leaf(fqdn)), fqdn


def test_pubrev008_safe_hmc_atoms_have_minimal_gaussian_behavior() -> None:
    from sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc import (
        initializehmcstate,
        leapfrogproposalkernel,
    )

    state, kernel = initializehmcstate(
        _standard_normal_logp,
        np.array([0.25, -0.5]),
        step_size=0.05,
        n_leapfrog=3,
        seed=11,
    )

    assert state.shape == (6,)
    assert kernel.tolist() == [0.05, 3.0, 2.0]
    assert np.allclose(state[:2], [0.25, -0.5])
    assert np.isclose(state[2], _standard_normal_logp(np.array([0.25, -0.5])))
    assert np.allclose(state[3:5], [-0.25, 0.5], atol=1e-5)

    proposal_in = np.array([0.25, -0.5, 0.1, -0.2])
    proposal_out = leapfrogproposalkernel(proposal_in, kernel, _standard_normal_logp)

    assert proposal_out.shape == (7,)
    assert np.all(np.isfinite(proposal_out))
    assert np.isclose(proposal_out[2], _standard_normal_logp(proposal_out[:2]))


def test_pubrev008_safe_hmc_llm_transition_threads_rng_deterministically() -> None:
    from sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc_llm import (
        hamiltoniantransitionkernel,
        initializehmckernelstate,
        initializesamplerrng,
    )

    kernel_spec, chain_state = initializehmckernelstate(
        _standard_normal_logp,
        np.array([0.1, -0.2]),
        step_size=0.03,
        n_leapfrog=4,
    )
    key = initializesamplerrng(17)

    state_a, key_a, stats_a = hamiltoniantransitionkernel(
        chain_state,
        kernel_spec,
        key,
        _standard_normal_logp,
    )
    state_b, key_b, stats_b = hamiltoniantransitionkernel(
        chain_state,
        kernel_spec,
        key,
        _standard_normal_logp,
    )

    assert state_a.shape == chain_state.shape
    assert key_a.shape == (1,)
    assert set(stats_a) == {"accepted", "accept_prob", "delta_H"}
    assert np.allclose(state_a, state_b)
    assert np.array_equal(key_a, key_b)
    assert all(np.allclose(stats_a[name], stats_b[name]) for name in stats_a)
    assert 0.0 <= float(stats_a["accept_prob"]) <= 1.0


def test_pubrev008_safe_nuts_initializer_accepts_vector_positions() -> None:
    from sciona.atoms.inference.mcmc_foundational.mini_mcmc.nuts_llm import initializenutsstate

    state, key = initializenutsstate(
        _standard_normal_logp,
        np.array([0.2, -0.1]),
        target_accept_p=0.8,
        seed=23,
    )

    assert state.shape == (7,)
    assert np.allclose(state[:2], [0.2, -0.1])
    assert np.isclose(state[2], _standard_normal_logp(np.array([0.2, -0.1])))
    assert np.allclose(state[3:5], [-0.2, 0.1], atol=1e-5)
    assert np.allclose(state[-2:], [0.8, 0.1])
    assert key.dtype == np.int64
    assert key.tolist() == [23]


def test_pubrev008_review_bundle_only_promotes_safe_mini_mcmc_rows() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    row_keys = {row["atom_key"] for row in bundle["rows"]}

    assert SAFE_ATOMS <= row_keys
    assert row_keys.isdisjoint(HELD_ATOMS)

    for row in bundle["rows"]:
        if row["atom_key"] not in SAFE_ATOMS:
            continue
        assert row["review_status"] == "reviewed"
        assert row["review_semantic_verdict"] == "pass"
        assert row["review_developer_semantic_verdict"] == "pass_with_limits"
        assert row["trust_readiness"] == "catalog_ready"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert isinstance(row["risk_score"], int)
        assert isinstance(row["acceptability_score"], int)
        assert row["acceptability_band"] == "review_ready"
        for rel_path in row["source_paths"]:
            assert (ROOT / rel_path).exists(), rel_path


def test_pubrev008_review_bundle_is_mergeable_without_unresolved_atoms() -> None:
    summary = merge_audit_manifest_with_review_bundles(
        manifest_path=ROOT / "data" / "audit_manifest.json",
        review_bundle_paths=[BUNDLE_PATH],
        dry_run=True,
    )

    assert summary["skipped_unresolved_atom_count"] == 0


def test_pubrev008_references_use_current_fqdns_and_registry_ids() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_ids = set(registry["references"])
    seen_fqdns: set[str] = set()

    for path in REFERENCE_PATHS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for atom_ref, entry in payload["atoms"].items():
            fqdn, _, source_ref = atom_ref.partition("@")
            seen_fqdns.add(fqdn)
            assert fqdn.startswith("sciona.atoms.inference.mcmc_foundational.mini_mcmc.")
            assert source_ref.startswith("src/sciona/atoms/inference/mcmc_foundational/mini_mcmc/")
            assert entry["references"]
            for ref in entry["references"]:
                assert ref["ref_id"] in registry_ids
                metadata = ref["match_metadata"]
                assert metadata["match_type"] == "manual"
                assert metadata["confidence"] in {"low", "medium", "high"}
                assert metadata["notes"]

    assert SAFE_ATOMS <= seen_fqdns
    assert HELD_ATOMS <= seen_fqdns
