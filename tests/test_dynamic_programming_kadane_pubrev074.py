from __future__ import annotations

import importlib
import json
from pathlib import Path

import icontract
import pytest

from sciona.atoms.audit_review_bundles import (
    VALID_ACCEPTABILITY_BANDS,
    VALID_PARITY_COVERAGE_LEVELS,
    merge_audit_manifest_with_review_bundles,
)


ROOT = Path(__file__).resolve().parents[1]
ATOM_FQDN = "sciona.atoms.dynamic_programming.kadane.max_subarray"
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "dynamic_programming_kadane_pubrev_074.review_bundle.json"
CDG_PATH = ROOT / "src" / "sciona" / "atoms" / "dynamic_programming" / "kadane" / "cdg.json"
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "dynamic_programming" / "kadane" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"
MANIFEST_PATH = ROOT / "data" / "audit_manifest.json"


def _bundle() -> dict:
    return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))


def test_pubrev074_kadane_behavior_matches_maximum_subarray_contract() -> None:
    from sciona.atoms.dynamic_programming.kadane import max_subarray

    assert max_subarray([1, 2, 3, 4]) == 10
    assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
    assert max_subarray([-3, -5, -2, -9]) == -2
    assert max_subarray([42]) == 42
    assert max_subarray([-42]) == -42
    assert max_subarray([1, 2, -3, 4, 5, -7, 23]) == 25


def test_pubrev074_kadane_rejects_empty_array() -> None:
    from sciona.atoms.dynamic_programming.kadane import max_subarray

    with pytest.raises(icontract.errors.ViolationError, match="array must not be empty"):
        max_subarray([])


def test_pubrev074_imports_provider_owned_runtime_from_cs_repo() -> None:
    module = importlib.import_module("sciona.atoms.dynamic_programming.kadane.atoms")

    assert module.max_subarray([5, -10, 7]) == 7
    assert module.__file__ is not None
    assert "sciona-atoms-cs" in module.__file__
    assert module.__file__.endswith("src/sciona/atoms/dynamic_programming/kadane/atoms.py")


def test_pubrev074_review_bundle_is_catalog_ready_and_db_compatible() -> None:
    bundle = _bundle()

    assert bundle["provider_repo"] == "sciona-atoms"
    assert bundle["family_batch"] == "dynamic_programming_kadane_pubrev_074"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass"
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] == "catalog_ready"
    assert bundle["blocking_findings"] == []
    assert bundle["required_actions"] == []

    rows = bundle["rows"]
    assert [row["atom_key"] for row in rows] == [ATOM_FQDN]
    row = rows[0]
    assert row["atom_name"] == ATOM_FQDN
    assert row["review_status"] == "reviewed"
    assert row["trust_readiness"] == "catalog_ready"
    assert row["overall_verdict"] == "acceptable_with_limits"
    assert row["structural_status"] == "pass"
    assert row["semantic_status"] == "pass"
    assert row["runtime_status"] == "pass"
    assert row["developer_semantics_status"] == "pass_with_limits"
    assert row["has_references"] is True
    assert row["references_status"] == "pass"
    assert isinstance(row["risk_score"], int)
    assert isinstance(row["acceptability_score"], int)
    assert row["acceptability_band"] in VALID_ACCEPTABILITY_BANDS
    assert row["parity_coverage_level"] in VALID_PARITY_COVERAGE_LEVELS
    assert row["parity_test_status"] == "pass"
    assert row["blocking_findings"] == []
    assert row["required_actions"] == []
    for rel_path in row["source_paths"]:
        assert (ROOT / rel_path).exists(), rel_path


def test_pubrev074_cdg_supports_io_specs_for_canonical_fqdn() -> None:
    cdg = json.loads(CDG_PATH.read_text(encoding="utf-8"))
    atomic_nodes = [node for node in cdg["nodes"] if node.get("status") == "atomic"]

    assert len(atomic_nodes) == 1
    node = atomic_nodes[0]
    assert node["node_id"] == "max_subarray"
    assert node["name"] == "max_subarray"
    assert node["inputs"] == [
        {
            "name": "array",
            "type_desc": "list[int]",
            "constraints": "non-empty",
            "data_kind": "",
            "time_basis": "",
            "provenance": "",
            "required": True,
            "default_value_repr": "",
        }
    ]
    assert node["outputs"] == [
        {
            "name": "result",
            "type_desc": "int",
            "constraints": "maximum contiguous subarray sum",
            "data_kind": "",
            "time_basis": "",
            "provenance": "",
            "required": True,
            "default_value_repr": "",
        }
    ]


def test_pubrev074_references_use_existing_registry_entry() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_ids = set(registry["references"])

    assert set(payload["atoms"]) == {ATOM_FQDN}
    refs = payload["atoms"][ATOM_FQDN]["references"]
    assert [ref["ref_id"] for ref in refs] == ["repo_keon_algorithms"]
    assert refs[0]["ref_id"] in registry_ids
    metadata = refs[0]["match_metadata"]
    assert metadata["match_type"] == "manual"
    assert metadata["confidence"] == "high"
    assert metadata["notes"]


def test_pubrev074_review_bundle_is_mergeable_without_unresolved_atoms() -> None:
    summary = merge_audit_manifest_with_review_bundles(
        manifest_path=MANIFEST_PATH,
        review_bundle_paths=[BUNDLE_PATH],
        dry_run=True,
    )

    assert summary["bundle_entry_count"] == 1
    assert summary["skipped_unresolved_atom_count"] == 0


def test_pubrev074_manifest_entry_is_publishable_after_bundle_merge() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    row = next(atom for atom in manifest["atoms"] if atom["atom_name"] == ATOM_FQDN)

    assert row["review_status"] == "approved"
    assert row["review_priority"] == "review_now"
    assert row["structural_status"] == "pass"
    assert row["semantic_status"] == "pass"
    assert row["runtime_status"] == "pass"
    assert row["developer_semantics_status"] == "pass"
    assert row["has_references"] is True
    assert row["references_status"] == "pass"
    assert row["argument_names"] == ["array"]
    assert row["argument_details"][0]["annotation"] == "list[int]"
    assert row["return_annotation"] == "int"
    assert row["docstring_summary"].startswith("Find the maximum sum of a contiguous subarray")
    assert row["risk_score"] == 18
    assert row["acceptability_score"] == 88
    assert row["acceptability_band"] == "review_ready"
    assert row["parity_coverage_level"] == "positive_and_negative"
