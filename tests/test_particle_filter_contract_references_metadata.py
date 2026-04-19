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

    expected_key = (
        "sciona.atoms.state_estimation.particle_filters.track_particle_hidden_state"
        "@sciona/atoms/state_estimation/particle_filters/atoms.py:37"
    )
    assert set(payload["atoms"]) == {expected_key}
    assert payload["atoms"][expected_key]["references"][0]["ref_id"] == "gordon1993particle"

    fqdn, _, _ = expected_key.partition("@")
    leaf = fqdn.removeprefix("sciona.atoms.state_estimation.particle_filters.")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    assert leaf in registered
