from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


def test_kalman_filter_contract_reference_key_is_canonical_and_registered() -> None:
    import_module("sciona.atoms.state_estimation.kalman_filters.atoms")

    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (
            root
            / "src"
            / "sciona"
            / "atoms"
            / "state_estimation"
            / "kalman_filters"
            / "references.json"
        ).read_text(encoding="utf-8")
    )

    expected_key = (
        "sciona.atoms.state_estimation.kalman_filters.track_linear_gaussian_state"
        "@sciona/atoms/state_estimation/kalman_filters/atoms.py:41"
    )
    assert set(payload["atoms"]) == {expected_key}

    fqdn, _, _ = expected_key.partition("@")
    leaf = fqdn.removeprefix("sciona.atoms.state_estimation.kalman_filters.")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    assert leaf in registered

