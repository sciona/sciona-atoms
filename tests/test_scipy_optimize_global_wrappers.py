from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import scipy.optimize

from sciona.atoms.scipy import optimize as optimize_atoms
from sciona.atoms.scipy.witnesses import (
    witness_differentialevolutionoptimization,
    witness_shgoglobaloptimization,
)
from sciona.ghost.abstract import AbstractArray


def _parameter_names(function: object) -> list[str]:
    return list(inspect.signature(function).parameters)


def test_global_optimizer_wrappers_match_installed_scipy_parameter_names() -> None:
    assert _parameter_names(optimize_atoms.shgo) == _parameter_names(scipy.optimize.shgo)
    assert _parameter_names(optimize_atoms.differential_evolution) == _parameter_names(
        scipy.optimize.differential_evolution
    )

    assert inspect.signature(optimize_atoms.shgo).parameters["workers"].kind is inspect.Parameter.KEYWORD_ONLY
    differential_signature = inspect.signature(optimize_atoms.differential_evolution)
    assert differential_signature.parameters["integrality"].kind is inspect.Parameter.KEYWORD_ONLY
    assert differential_signature.parameters["vectorized"].kind is inspect.Parameter.KEYWORD_ONLY
    assert differential_signature.parameters["seed"].kind is inspect.Parameter.KEYWORD_ONLY


def test_shgo_wrapper_forwards_current_scipy_keywords(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_shgo(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(optimize_atoms.scipy.optimize, "shgo", fake_shgo)

    result = optimize_atoms.shgo(lambda x: float(x[0] ** 2), [(-1.0, 1.0)], workers=1)

    assert result is sentinel
    assert captured["args"][1] == [(-1.0, 1.0)]
    kwargs = captured["kwargs"]
    assert kwargs["constraints"] is None
    assert kwargs["sampling_method"] == "simplicial"
    assert kwargs["workers"] == 1


def test_differential_evolution_wrapper_forwards_current_scipy_keywords(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()
    rng = np.random.default_rng(123)
    integrality = np.array([True])

    def fake_differential_evolution(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        optimize_atoms.scipy.optimize,
        "differential_evolution",
        fake_differential_evolution,
    )

    result = optimize_atoms.differential_evolution(
        lambda x: float(x[0] ** 2),
        [(-1.0, 1.0)],
        rng=rng,
        integrality=integrality,
        seed=None,
        vectorized=False,
    )

    assert result is sentinel
    assert captured["args"][1] == [(-1.0, 1.0)]
    kwargs = captured["kwargs"]
    assert kwargs["rng"] is rng
    assert kwargs["integrality"] is integrality
    assert "seed" not in kwargs
    assert kwargs["vectorized"] is False


def test_differential_evolution_wrapper_preserves_legacy_seed_forwarding(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_differential_evolution(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        optimize_atoms.scipy.optimize,
        "differential_evolution",
        fake_differential_evolution,
    )

    result = optimize_atoms.differential_evolution(
        lambda x: float(x[0] ** 2),
        [(-1.0, 1.0)],
        seed=123,
    )

    assert result is sentinel
    kwargs = captured["kwargs"]
    assert kwargs["seed"] == 123
    assert "rng" not in kwargs


def test_global_optimizer_wrappers_execute_small_scipy_problems() -> None:
    objective = lambda x: float((x[0] - 0.25) ** 2)

    shgo_result = optimize_atoms.shgo(objective, [(-1.0, 1.0)], n=8, iters=1)
    rng_result = optimize_atoms.differential_evolution(
        objective,
        [(-1.0, 1.0)],
        maxiter=1,
        popsize=5,
        polish=False,
        rng=1,
    )
    seed_result = optimize_atoms.differential_evolution(
        objective,
        [(-1.0, 1.0)],
        maxiter=1,
        popsize=5,
        polish=False,
        seed=1,
    )

    assert shgo_result.x.shape == (1,)
    assert rng_result.x.shape == (1,)
    assert seed_result.x.shape == (1,)
    assert np.isfinite(shgo_result.fun)
    assert np.isfinite(rng_result.fun)
    assert np.isfinite(seed_result.fun)


def test_global_optimizer_witnesses_describe_solution_vector_shape() -> None:
    bounds = AbstractArray(shape=(2, 2), dtype="float64")

    shgo_meta = witness_shgoglobaloptimization(lambda x: 0.0, bounds)
    differential_meta = witness_differentialevolutionoptimization(lambda x: 0.0, bounds)

    assert shgo_meta.shape == (2,)
    assert differential_meta.shape == (2,)
    assert shgo_meta.dtype == "float64"
    assert differential_meta.dtype == "float64"


def test_scipy_optimize_pubrev_026_bundle_ratchets_only_global_remainder() -> None:
    bundle_path = Path("data/review_bundles/scipy_optimize_pubrev_026.review_bundle.json")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    assert bundle["family_batch"] == "scipy_optimize_pubrev_026"
    rows = {row["atom_name"]: row for row in bundle["rows"]}
    assert set(rows) == {
        "sciona.atoms.scipy.optimize.differential_evolution",
        "sciona.atoms.scipy.optimize.shgo",
    }
    for row in rows.values():
        assert row["trust_readiness"] == "catalog_ready"
        assert row["overall_verdict"] == "acceptable_with_limits"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["parity_test_status"] == "pass"
        assert not row["blocking_findings"]
        assert not row["required_actions"]
