from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = (
    ROOT
    / "src"
    / "sciona"
    / "atoms"
    / "inference"
    / "belief_propagation"
    / "loopy_bp"
    / "references.json"
)


def test_inference_belief_propagation_reference_keys_are_canonical() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))

    atom_keys = sorted(payload["atoms"])
    assert atom_keys == [
        "sciona.atoms.inference.belief_propagation.loopy_bp.initialize_message_passing_state@sciona/atoms/inference/belief_propagation/loopy_bp/atoms.py:26",
        "sciona.atoms.inference.belief_propagation.loopy_bp.run_loopy_message_passing_and_belief_query@sciona/atoms/inference/belief_propagation/loopy_bp/atoms.py:56",
    ]
    assert not any(key.startswith("sciona.atoms.belief_propagation.") for key in atom_keys)
