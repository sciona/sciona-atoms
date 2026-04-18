from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


def test_belief_propagation_reference_keys_are_canonical_and_registered() -> None:
    import_module("sciona.atoms.expansion.belief_propagation")

    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (
            root
            / "src"
            / "sciona"
            / "atoms"
            / "expansion"
            / "belief_propagation"
            / "references.json"
        ).read_text(encoding="utf-8")
    )

    prefix = "sciona.atoms.expansion.belief_propagation."
    atom_keys = sorted(payload["atoms"])
    assert atom_keys
    assert all(key.startswith(prefix) for key in atom_keys)

    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    for key in atom_keys:
        fqdn, _, _ = key.partition("@")
        leaf = fqdn.removeprefix(prefix)
        assert leaf in registered
        assert leaf
