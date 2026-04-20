from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _standard_normal_logp(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    return float(-0.5 * np.sum(x**2))


def test_kthohr_aees_transition_uses_explicit_state_and_rng() -> None:
    from sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.aees import (
        metropolishastingstransitionkernel,
    )

    state = np.array([0.25, -0.5])
    key = np.array([123], dtype=np.int64)
    out_a, key_a = metropolishastingstransitionkernel(state, 2.0, _standard_normal_logp, key)
    out_b, key_b = metropolishastingstransitionkernel(state, 2.0, _standard_normal_logp, key)

    assert out_a.shape == state.shape
    assert key_a.shape == key.shape
    assert np.allclose(out_a, out_b)
    assert np.array_equal(key_a, key_b)
    assert np.allclose(state, [0.25, -0.5])


def test_kthohr_aees_target_oracle_matches_gaussian_mixture_example() -> None:
    from sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.aees import targetlogkerneloracle

    weights = np.array([0.5, 0.5])
    means = np.array([[-2.0, -2.0], [2.0, 2.0]])
    variances = np.array([0.1, 0.1])
    value = targetlogkerneloracle(np.array([-2.0, -2.0]), weights, means, variances)
    expected_at_mode = np.log(0.5) - np.log(2.0 * np.pi * 0.1)

    assert np.isclose(value, expected_at_mode, atol=1e-12)


def test_kthohr_rwmh_and_hmc_kernels_are_deterministic_for_same_key() -> None:
    from sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.hmc import (
        buildhmckernelfromlogdensityoracle,
    )
    from sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.rwmh import (
        constructrandomwalkmetropoliskernel,
    )

    state = np.array([0.1, -0.2])
    key = np.array([11], dtype=np.int64)
    rwmh = constructrandomwalkmetropoliskernel(_standard_normal_logp, proposal_scale=0.2)
    hmc = buildhmckernelfromlogdensityoracle(_standard_normal_logp, step_size=0.03, n_leapfrog=3)

    rwmh_state_a, rwmh_key_a = rwmh(state, key)
    rwmh_state_b, rwmh_key_b = rwmh(state, key)
    hmc_state_a, hmc_key_a = hmc(state, key)
    hmc_state_b, hmc_key_b = hmc(state, key)

    assert np.allclose(rwmh_state_a, rwmh_state_b)
    assert np.array_equal(rwmh_key_a, rwmh_key_b)
    assert np.allclose(hmc_state_a, hmc_state_b)
    assert np.array_equal(hmc_key_a, hmc_key_b)
    assert np.all(np.isfinite(hmc_state_a))


def test_kthohr_de_kernel_requires_population_and_threads_key() -> None:
    from sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.de import build_de_transition_kernel

    population = np.array([[0.1, -0.2], [0.4, 0.3], [-0.3, 0.2], [0.0, 0.5]])
    key = np.array([19], dtype=np.int64)
    kernel = build_de_transition_kernel(_standard_normal_logp)

    out_a, key_a = kernel(population, key)
    out_b, key_b = kernel(population, key)

    assert out_a.shape == population.shape
    assert np.allclose(out_a, out_b)
    assert np.array_equal(key_a, key_b)
    assert np.all(np.isfinite(out_a))


def test_kthohr_mala_adjustment_matches_unbounded_log_ratio() -> None:
    from sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.mala import mala_proposal_adjustment

    def mean_fn(values: np.ndarray, step_size: float) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) - 0.5 * step_size**2 * np.asarray(values)

    prop = np.array([0.2, -0.1])
    prev = np.array([0.3, -0.2])
    step = 0.4
    precond = np.eye(2)
    adjustment = mala_proposal_adjustment(prop, prev, step, precond, mean_fn)

    assert isinstance(adjustment, float)
    assert np.isfinite(adjustment)
    assert not np.isclose(adjustment, 0.0)


def test_kthohr_dispatch_uses_callable_target_not_proxy_density_array() -> None:
    from sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.mcmc_algos import (
        dispatch_mcmc_algorithm,
    )

    key = np.array([29], dtype=np.int64)
    rwmh_samples = dispatch_mcmc_algorithm("rwmh", _standard_normal_logp, np.array([0.1, -0.2]), 4, key)
    hmc_samples = dispatch_mcmc_algorithm("hmc", _standard_normal_logp, np.array([0.1, -0.2]), 3, key)
    de_samples = dispatch_mcmc_algorithm(
        "de",
        _standard_normal_logp,
        np.array([[0.1, -0.2], [0.4, 0.3], [-0.3, 0.2]]),
        2,
        key,
    )

    assert rwmh_samples.shape == (4, 2)
    assert hmc_samples.shape == (3, 2)
    assert de_samples.shape == (2, 3, 2)
    assert np.all(np.isfinite(rwmh_samples))
    assert np.all(np.isfinite(hmc_samples))
    assert np.all(np.isfinite(de_samples))


def test_kthohr_repaired_metadata_uses_canonical_fqdns() -> None:
    bundle = json.loads((ROOT / "data" / "review_bundles" / "mcmc_foundational.review_bundle.json").read_text())
    flat_refs = json.loads(
        (
            ROOT
            / "src"
            / "sciona"
            / "atoms"
            / "inference"
            / "mcmc_foundational"
            / "kthohr_mcmc"
            / "references.json"
        ).read_text()
    )
    aees_refs = json.loads(
        (
            ROOT
            / "src"
            / "sciona"
            / "atoms"
            / "inference"
            / "mcmc_foundational"
            / "kthohr_mcmc"
            / "aees"
            / "references.json"
        ).read_text()
    )
    rows = {
        row["atom_key"]: row
        for row in bundle["rows"]
        if row.get("atom_key", "").startswith("sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.")
    }

    assert "sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.hmc.buildhmckernelfromlogdensityoracle" in rows
    assert rows["sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.nuts.nuts_recursive_tree_build"]["trust_readiness"] == "remediation_hold"
    assert all(key.startswith("sciona.atoms.inference.") for key in flat_refs["atoms"])
    assert all(key.startswith("sciona.atoms.inference.") for key in aees_refs["atoms"])


def test_kthohr_flat_module_cdg_derives_module_qualified_fqdn() -> None:
    from sciona.atoms.supabase_backfill import derive_atom_fqdn

    atoms_root = ROOT / "src" / "sciona" / "atoms"
    cdg_path = atoms_root / "inference" / "mcmc_foundational" / "kthohr_mcmc" / "de_cdg.json"

    assert (
        derive_atom_fqdn(cdg_path, atoms_root, "build_de_transition_kernel")
        == "sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.de.build_de_transition_kernel"
    )
