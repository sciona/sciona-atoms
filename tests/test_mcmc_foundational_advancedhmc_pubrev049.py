from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from sciona.atoms.audit_review_bundles import merge_audit_manifest_with_review_bundles


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "advancedhmc_pubrev_049.review_bundle.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"
INTEGRATOR_REFS_PATH = (
    ROOT
    / "src"
    / "sciona"
    / "atoms"
    / "inference"
    / "mcmc_foundational"
    / "advancedhmc"
    / "integrator"
    / "references.json"
)
TRAJECTORY_REFS_PATH = (
    ROOT
    / "src"
    / "sciona"
    / "atoms"
    / "inference"
    / "mcmc_foundational"
    / "advancedhmc"
    / "trajectory"
    / "references.json"
)

TARGET_ATOMS = {
    "sciona.atoms.inference.mcmc_foundational.advancedhmc.integrator.hamiltonianphasepointtransition",
    "sciona.atoms.inference.mcmc_foundational.advancedhmc.integrator.temperingfactorcomputation",
    "sciona.atoms.inference.mcmc_foundational.advancedhmc.trajectory.buildnutstree",
    "sciona.atoms.inference.mcmc_foundational.advancedhmc.trajectory.nutstransitionkernel",
}


def _import_leaf(fqdn: str):
    module_name, _, symbol_name = fqdn.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _quadratic_potential(x: np.ndarray) -> float:
    values = np.asarray(x, dtype=np.float64)
    return float(0.5 * np.dot(values, values))


def test_pubrev049_target_atoms_are_importable() -> None:
    for fqdn in sorted(TARGET_ATOMS):
        assert callable(_import_leaf(fqdn)), fqdn


def test_pubrev049_tempering_matches_advancedhmc_half_step_schedule() -> None:
    from sciona.atoms.inference.mcmc_foundational.advancedhmc.integrator import (
        temperingfactorcomputation,
    )

    lf = np.array([0.1, 4.0])
    r = np.ones(3)

    assert temperingfactorcomputation(lf, r, {"i": 1, "is_half": True}, 3) == 2.0
    assert temperingfactorcomputation(lf, r, {"i": 1, "is_half": False}, 3) == 2.0
    assert temperingfactorcomputation(lf, r, {"i": 2, "is_half": True}, 3) == 2.0
    assert temperingfactorcomputation(lf, r, {"i": 2, "is_half": False}, 3) == 0.5
    assert temperingfactorcomputation(lf, r, {"i": 3, "is_half": True}, 3) == 0.5
    assert temperingfactorcomputation(lf, r, {"i": 3, "is_half": False}, 3) == 0.5

    with pytest.raises(IndexError):
        temperingfactorcomputation(lf, r, {"i": 4, "is_half": False}, 3)


def test_pubrev049_phasepoint_transition_is_finite_leapfrog_step() -> None:
    from sciona.atoms.inference.mcmc_foundational.advancedhmc.integrator import (
        hamiltonianphasepointtransition,
    )

    z_next, is_valid = hamiltonianphasepointtransition(
        np.array([0.1, 1.0]),
        _quadratic_potential,
        np.array([1.0, 0.0]),
        1.0,
    )

    assert is_valid is True
    assert z_next.shape == (2,)
    assert np.all(np.isfinite(z_next))
    assert np.allclose(z_next, np.array([0.995, -0.09975]), atol=1e-6)


def test_pubrev049_build_nuts_tree_returns_directional_compact_leaves() -> None:
    from sciona.atoms.inference.mcmc_foundational.advancedhmc.trajectory import buildnutstree

    start = np.array([0.0, 1.0])
    initial_energy = _quadratic_potential(start[:1]) + 0.5 * np.dot(start[1:], start[1:])
    forward = buildnutstree(np.array([7]), _quadratic_potential, start, 1, 2, initial_energy)
    backward = buildnutstree(np.array([7]), _quadratic_potential, start, -1, 2, initial_energy)

    assert forward.ndim == 2
    assert forward.shape[1] == 2
    assert 1 <= forward.shape[0] <= 4
    assert np.all(np.isfinite(forward))
    assert forward[0, 0] > 0.0
    assert backward[0, 0] < 0.0

    with pytest.raises(ValueError):
        buildnutstree(np.array([7]), _quadratic_potential, start, 0, 2, initial_energy)


def test_pubrev049_nuts_transition_is_deterministic_for_key_and_reports_stats() -> None:
    from sciona.atoms.inference.mcmc_foundational.advancedhmc.trajectory import nutstransitionkernel

    initial = np.array([0.25, -0.5])
    params = np.array([0.05, 4.0, 1000.0])
    key = np.array([17])

    state_a, stats_a = nutstransitionkernel(key, _quadratic_potential, initial, params)
    state_b, stats_b = nutstransitionkernel(key, _quadratic_potential, initial, params)

    assert state_a.shape == initial.shape
    assert np.all(np.isfinite(state_a))
    assert np.allclose(state_a, state_b)
    assert set(stats_a) == {"accept_prob", "tree_depth", "energy", "n_steps", "numerical_error"}
    assert all(np.allclose(stats_a[name], stats_b[name]) for name in stats_a)
    assert 0.0 <= float(stats_a["accept_prob"]) <= 1.0
    assert 0 <= int(stats_a["tree_depth"]) <= 4
    assert int(stats_a["n_steps"]) >= 0
    assert float(stats_a["numerical_error"]) in {0.0, 1.0}


def test_pubrev049_references_use_live_fqdns_and_registered_ids() -> None:
    registry_ids = set(json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"])
    seen_fqdns: set[str] = set()

    for path in (INTEGRATOR_REFS_PATH, TRAJECTORY_REFS_PATH):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for atom_key, entry in payload["atoms"].items():
            seen_fqdns.add(atom_key)
            assert atom_key in TARGET_ATOMS
            assert "@" not in atom_key
            assert ".mcmc_foundational.advancedhmc." in atom_key
            assert ".inference." in atom_key
            assert entry["references"]
            for reference in entry["references"]:
                assert reference["ref_id"] in registry_ids
                metadata = reference["match_metadata"]
                assert metadata["match_type"] == "manual"
                assert metadata["confidence"] in {"medium", "high"}
                assert metadata["matched_nodes"]
                assert metadata["notes"]

    assert seen_fqdns == TARGET_ATOMS


def test_pubrev049_review_bundle_covers_only_target_atoms_and_is_mergeable() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    row_keys = {row["atom_key"] for row in bundle["rows"]}

    assert bundle["review_record_path"] == "data/review_bundles/advancedhmc_pubrev_049.review_bundle.json"
    assert row_keys == TARGET_ATOMS
    for row in bundle["rows"]:
        assert row["review_status"] == "reviewed"
        assert row["review_semantic_verdict"] in {"pass", "pass_with_limits"}
        assert row["review_developer_semantic_verdict"] == "pass_with_limits"
        assert row["trust_readiness"] == "catalog_ready"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["blocking_findings"] == []
        assert row["required_actions"] == []
        assert row["acceptability_band"] == "review_ready"
        for rel_path in row["source_paths"]:
            assert (ROOT / rel_path).exists(), rel_path

    summary = merge_audit_manifest_with_review_bundles(
        manifest_path=ROOT / "data" / "audit_manifest.json",
        review_bundle_paths=[BUNDLE_PATH],
        dry_run=True,
    )
    assert summary["bundle_entry_count"] == 4
    assert summary["skipped_unresolved_atom_count"] == 0
