from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


def test_particle_filter_contract_reference_key_is_canonical_and_registered() -> None:
    import_module("sciona.atoms.state_estimation.particle_filters.atoms")

    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (
            root
            / "src"
            / "sciona"
            / "atoms"
            / "state_estimation"
            / "particle_filters"
            / "references.json"
        ).read_text(encoding="utf-8")
    )

    expected_tracker_key = (
        "sciona.atoms.state_estimation.particle_filters.track_particle_hidden_state"
        "@sciona/atoms/state_estimation/particle_filters/atoms.py:37"
    )
    expected_basic_keys = {
        "sciona.atoms.state_estimation.particle_filters.basic.filter_step_preparation_and_dispatch",
        "sciona.atoms.state_estimation.particle_filters.basic.hypothesis_propagation_kernel",
        "sciona.atoms.state_estimation.particle_filters.basic.likelihood_reweight_kernel",
        "sciona.atoms.state_estimation.particle_filters.basic.resample_and_hypothesis_distribution_projection",
    }
    assert {expected_tracker_key, *expected_basic_keys}.issubset(payload["atoms"])
    assert payload["atoms"][expected_tracker_key]["references"][0]["ref_id"] == "gordon1993particle"
    for key in expected_basic_keys:
        assert {ref["ref_id"] for ref in payload["atoms"][key]["references"]} == {
            "gordon1993particle",
            "repo_particlefilters_jl",
        }

    fqdn, _, _ = expected_tracker_key.partition("@")
    leaf = fqdn.removeprefix("sciona.atoms.state_estimation.particle_filters.")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    assert leaf in registered
