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
    / "causal_inference"
    / "feature_primitives"
    / "references.json"
)
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.causal_inference.feature_primitives.igci_asymmetry_score",
    "sciona.atoms.causal_inference.feature_primitives.hsic_independence_test",
    "sciona.atoms.causal_inference.feature_primitives.knn_entropy_estimator",
    "sciona.atoms.causal_inference.feature_primitives.uniform_divergence",
    "sciona.atoms.causal_inference.feature_primitives.normalized_error_probability",
    "sciona.atoms.causal_inference.feature_primitives.discretize_and_bin",
    "sciona.atoms.causal_inference.feature_primitives.polyfit_nonlinearity_asymmetry",
    "sciona.atoms.causal_inference.feature_primitives.polyfit_residual_error",
}


def test_references_json_exists_and_has_eight_fqdns() -> None:
    assert REFERENCES_PATH.exists()
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    atom_keys = set(payload["atoms"])
    assert len(atom_keys) == 8
    fqdns = {k.partition("@")[0] for k in atom_keys}
    assert fqdns == EXPECTED_FQDNS


def test_each_atom_has_nonempty_references() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for key, entry in payload["atoms"].items():
        assert entry["references"], f"empty references for {key}"


def test_ref_ids_exist_in_registry() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_ids = set(registry["references"])
    for key, entry in payload["atoms"].items():
        for ref in entry["references"]:
            assert ref["ref_id"] in registry_ids, f"{ref['ref_id']} not in registry (atom {key})"


def test_each_reference_has_match_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for key, entry in payload["atoms"].items():
        for ref in entry["references"]:
            mm = ref["match_metadata"]
            assert "match_type" in mm, f"missing match_type for {ref['ref_id']} in {key}"
            assert "confidence" in mm, f"missing confidence for {ref['ref_id']} in {key}"
            assert "notes" in mm, f"missing notes for {ref['ref_id']} in {key}"


def test_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.causal_inference.feature_primitives.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    for fqdn in EXPECTED_FQDNS:
        leaf = fqdn.removeprefix("sciona.atoms.causal_inference.feature_primitives.")
        assert leaf in registered, f"{leaf} not in REGISTRY"
