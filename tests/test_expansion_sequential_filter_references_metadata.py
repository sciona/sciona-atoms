from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = (
    ROOT
    / "src"
    / "sciona"
    / "atoms"
    / "expansion"
    / "sequential_filter"
    / "references.json"
)
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"


def test_expansion_sequential_filter_reference_keys_are_canonical_registered_and_local() -> None:
    import_module("sciona.atoms.expansion.sequential_filter")

    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"]

    prefix = "sciona.atoms.expansion.sequential_filter."
    atom_keys = sorted(payload["atoms"])
    assert atom_keys
    assert all(key.startswith(prefix) for key in atom_keys)

    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    for key in atom_keys:
        fqdn, _, _ = key.partition("@")
        leaf = fqdn.removeprefix(prefix)
        assert leaf in registered
        assert leaf

        entry = payload["atoms"][key]
        assert entry["references"]
        for reference in entry["references"]:
            ref_id = reference["ref_id"]
            assert ref_id in registry
            assert reference["match_metadata"]["match_type"] == "manual"
            assert reference["match_metadata"]["confidence"] in {"high", "medium"}


def test_expansion_sequential_filter_references_cover_current_targets() -> None:
    from sciona.atoms.expansion.sequential_filter import SEQUENTIAL_FILTER_DECLARATIONS

    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    reference_fqdns = {key.partition("@")[0] for key in payload["atoms"]}
    declaration_fqdns = {fqdn for fqdn, _, _ in SEQUENTIAL_FILTER_DECLARATIONS.values()}

    assert reference_fqdns == declaration_fqdns
