from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _standard_normal_logp(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    return float(-0.5 * np.sum(x**2))


def _fd_grad(position: np.ndarray) -> np.ndarray:
    return -np.asarray(position, dtype=np.float64)


def _pack_phase(position: np.ndarray, momentum: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    momentum = np.asarray(momentum, dtype=np.float64)
    return np.concatenate([position, momentum, _fd_grad(position), [_standard_normal_logp(position)]])


def _test_leapfrog_phase(state: np.ndarray, step_size: float, direction_val: int) -> np.ndarray:
    vector = np.asarray(state, dtype=np.float64)
    dim = (vector.size - 1) // 3
    position = vector[:dim]
    momentum = vector[dim : 2 * dim]
    gradient = vector[2 * dim : 3 * dim]
    signed_step = float(direction_val) * float(step_size)

    momentum_half = momentum + 0.5 * signed_step * gradient
    position_next = position + signed_step * momentum_half
    gradient_next = _fd_grad(position_next)
    momentum_next = momentum_half + 0.5 * signed_step * gradient_next
    return np.concatenate([position_next, momentum_next, gradient_next, [_standard_normal_logp(position_next)]])


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


def test_kthohr_rmhmc_uses_metric_derivative_callback_and_threads_key() -> None:
    from sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.rmhmc import (
        buildrmhmctransitionkernel,
    )

    calls = {"target": 0, "tensor": 0}

    def target_with_grad(values: np.ndarray) -> tuple[float, np.ndarray]:
        calls["target"] += 1
        x = np.asarray(values, dtype=np.float64)
        return float(-0.5 * x @ x), -x

    def metric_with_derivative(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls["tensor"] += 1
        x = np.asarray(values, dtype=np.float64)
        scale = 1.0 + 0.1 * x[0] ** 2
        metric = scale * np.eye(x.size)
        derivative = np.zeros((x.size, x.size, x.size), dtype=np.float64)
        derivative[0] = 0.2 * x[0] * np.eye(x.size)
        return metric, derivative

    state = np.array([0.2, -0.15])
    key = np.array([41], dtype=np.int64)
    kernel = buildrmhmctransitionkernel(
        target_with_grad,
        metric_with_derivative,
        step_size=0.03,
        n_leapfrog=2,
        n_fixed_point=3,
    )

    state_a, key_a = kernel(state, key)
    state_b, key_b = kernel(state, key)

    assert state_a.shape == state.shape
    assert key_a.shape == key.shape
    assert np.allclose(state_a, state_b)
    assert np.array_equal(key_a, key_b)
    assert np.all(np.isfinite(state_a))
    assert calls["target"] > 0
    assert calls["tensor"] > 0
    assert calls["tensor"] <= 20


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


def test_kthohr_nuts_leaf_matches_source_slice_continue_and_acceptance_semantics() -> None:
    from sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.nuts import nuts_recursive_tree_build

    position = np.array([0.2, -0.1])
    momentum = np.array([0.3, 0.4])
    phase_state = _pack_phase(position, momentum)
    initial_joint = _standard_normal_logp(position) - 0.5 * float(np.dot(momentum, momentum))
    log_slice = initial_joint - 0.05

    tree = nuts_recursive_tree_build(
        1,
        0.05,
        log_slice,
        phase_state,
        _standard_normal_logp,
        _test_leapfrog_phase,
        0,
    )
    expected_state = _test_leapfrog_phase(phase_state, 0.05, 1)
    expected_position = expected_state[:2]
    expected_momentum = expected_state[2:4]
    expected_joint = _standard_normal_logp(expected_position) - 0.5 * float(np.dot(expected_momentum, expected_momentum))

    assert np.allclose(tree["position_minus"], expected_position)
    assert np.allclose(tree["position_plus"], expected_position)
    assert np.allclose(tree["position_proposal"], expected_position)
    assert np.allclose(tree["momentum_minus"], expected_momentum)
    assert np.allclose(tree["momentum_plus"], expected_momentum)
    assert int(tree["n_valid"]) == int(log_slice <= expected_joint)
    assert int(tree["should_continue"]) == int(log_slice < 1000.0 + expected_joint)
    assert int(tree["diverged"]) == int(log_slice >= 1000.0 + expected_joint)
    assert np.isclose(float(tree["alpha_sum"]), np.exp(min(0.0, expected_joint - initial_joint)))
    assert int(tree["n_alpha"]) == 1


def test_kthohr_nuts_recursive_tree_returns_source_shaped_bookkeeping() -> None:
    from sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.nuts import nuts_recursive_tree_build

    position = np.array([0.2, -0.1])
    momentum = np.array([0.3, 0.4])
    phase_state = _pack_phase(position, momentum)
    initial_joint = _standard_normal_logp(position) - 0.5 * float(np.dot(momentum, momentum))
    tree = nuts_recursive_tree_build(
        1,
        0.03,
        initial_joint - 0.1,
        phase_state,
        _standard_normal_logp,
        _test_leapfrog_phase,
        2,
        rng_key=np.array([101]),
    )

    expected_keys = {
        "position_minus",
        "momentum_minus",
        "grad_minus",
        "position_plus",
        "momentum_plus",
        "grad_plus",
        "position_proposal",
        "grad_proposal",
        "logp_proposal",
        "n_valid",
        "should_continue",
        "alpha_sum",
        "n_alpha",
        "diverged",
    }
    assert set(tree) == expected_keys
    assert tree["position_minus"].shape == position.shape
    assert tree["position_plus"].shape == position.shape
    assert tree["position_proposal"].shape == position.shape
    assert int(tree["n_alpha"]) == 4
    assert int(tree["n_valid"]) >= 1
    assert np.all(np.isfinite(tree["position_proposal"]))
    assert not np.allclose(tree["position_minus"], tree["position_plus"])


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
    assert rows["sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.nuts.nuts_recursive_tree_build"]["trust_readiness"] == "catalog_ready"
    assert rows["sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.rmhmc.buildrmhmctransitionkernel"]["trust_readiness"] == "catalog_ready"
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
