from __future__ import annotations

import json
from pathlib import Path

import icontract
import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.csgraph
import scipy.stats

from sciona.atoms.audit_review_bundles import (
    VALID_ACCEPTABILITY_BANDS,
    VALID_PARITY_COVERAGE_LEVELS,
    merge_audit_manifest_with_review_bundles,
)
from sciona.atoms.scipy import sparse_graph
from sciona.atoms.scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "scipy_remediation_pubrev_077.review_bundle.json"
CDG_PATH = ROOT / "src" / "sciona" / "atoms" / "scipy" / "cdg.json"
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "scipy" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"
MANIFEST_PATH = ROOT / "data" / "audit_manifest.json"

PROMOTED_ATOMS = {
    "sciona.atoms.scipy.sparse_graph.graph_laplacian",
    "sciona.atoms.scipy.sparse_graph.single_source_shortest_path",
    "sciona.atoms.scipy.sparse_graph.all_pairs_shortest_path",
    "sciona.atoms.scipy.sparse_graph.minimum_spanning_tree",
    "sciona.atoms.scipy.stats.norm",
}

HELD_ATOMS = {
    "sciona.atoms.scipy.sparse_graph.graph_fourier_transform",
    "sciona.atoms.scipy.sparse_graph.inverse_graph_fourier_transform",
    "sciona.atoms.scipy.sparse_graph.heat_kernel_diffusion",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dense(value: object) -> np.ndarray:
    if scipy.sparse.issparse(value):
        return value.toarray()
    return np.asarray(value)


def test_sparse_graph_wrappers_match_installed_scipy_behavior() -> None:
    graph = np.array(
        [
            [0.0, 2.0, 0.0, 1.0],
            [2.0, 0.0, 3.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
            [1.0, 0.0, 4.0, 0.0],
        ]
    )
    sparse = scipy.sparse.csr_matrix(graph)

    np.testing.assert_allclose(
        _dense(sparse_graph.graph_laplacian(sparse)),
        _dense(scipy.sparse.csgraph.laplacian(sparse, return_diag=False)),
    )
    np.testing.assert_allclose(
        sparse_graph.single_source_shortest_path(graph, indices=0, directed=False),
        scipy.sparse.csgraph.shortest_path(graph, directed=False, indices=0),
    )
    np.testing.assert_allclose(
        sparse_graph.all_pairs_shortest_path(graph, directed=False),
        scipy.sparse.csgraph.shortest_path(graph, directed=False, indices=None),
    )
    np.testing.assert_allclose(
        sparse_graph.minimum_spanning_tree(sparse).toarray(),
        scipy.sparse.csgraph.minimum_spanning_tree(sparse).toarray(),
    )


def test_sparse_graph_wrappers_cover_predecessor_and_sparse_cases() -> None:
    graph = scipy.sparse.csr_matrix(
        np.array(
            [
                [0.0, 5.0, 0.0],
                [5.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )

    distances, predecessors = sparse_graph.single_source_shortest_path(
        graph,
        indices=np.array([0, 1]),
        directed=False,
        return_predecessors=True,
    )
    expected_distances, expected_predecessors = scipy.sparse.csgraph.shortest_path(
        graph,
        directed=False,
        indices=np.array([0, 1]),
        return_predecessors=True,
    )

    np.testing.assert_allclose(distances, expected_distances)
    np.testing.assert_array_equal(predecessors, expected_predecessors)


def test_norm_wrapper_matches_frozen_scipy_normal_distribution() -> None:
    frozen = norm(loc=2.0, scale=3.0)
    expected = scipy.stats.norm(loc=2.0, scale=3.0)

    assert frozen.mean() == expected.mean()
    assert frozen.std() == expected.std()
    np.testing.assert_allclose(frozen.pdf([2.0, 5.0]), expected.pdf([2.0, 5.0]))
    np.testing.assert_allclose(frozen.cdf([-1.0, 2.0, 5.0]), expected.cdf([-1.0, 2.0, 5.0]))


def test_norm_wrapper_rejects_invalid_scale() -> None:
    with pytest.raises(icontract.errors.ViolationError, match="Scale must be positive"):
        norm(scale=0.0)


def test_pubrev077_review_bundle_promotes_only_source_aligned_remediation_subset() -> None:
    bundle = _load_json(BUNDLE_PATH)

    assert bundle["provider_repo"] == "sciona-atoms"
    assert bundle["family_batch"] == "scipy_remediation_pubrev_077"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass_with_limits"
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] == "catalog_ready"
    assert bundle["blocking_findings"] == []
    assert bundle["required_actions"] == []

    rows = bundle["rows"]
    row_keys = {row["atom_key"] for row in rows}
    assert row_keys == PROMOTED_ATOMS
    assert row_keys.isdisjoint(HELD_ATOMS)

    for row in rows:
        assert row["atom_name"] == row["atom_key"]
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


def test_pubrev077_cdg_has_io_specs_for_promoted_atoms_only() -> None:
    cdg = _load_json(CDG_PATH)
    nodes = {node["name"]: node for node in cdg["nodes"] if node.get("status") == "atomic"}

    assert set(nodes) == {fqdn.rsplit(".", 1)[-1] for fqdn in PROMOTED_ATOMS}
    for node in nodes.values():
        assert node["inputs"]
        assert node["outputs"] == [{"name": "result", "type_desc": node["outputs"][0]["type_desc"]}]
        assert all("required" in item for item in node["inputs"])

    assert set(cdg["metadata"]["held_atoms"]) == HELD_ATOMS


def test_pubrev077_references_bind_promoted_atoms_to_local_registry() -> None:
    references = _load_json(REFERENCES_PATH)
    registry = _load_json(REGISTRY_PATH)
    registry_ids = set(registry["references"])
    reference_fqdns = {key.split("@", 1)[0] for key in references["atoms"]}

    assert reference_fqdns == PROMOTED_ATOMS
    assert {"repo_scipy", "scipy2020"}.issubset(registry_ids)

    for atom_key, record in references["atoms"].items():
        assert "@" in atom_key
        assert record["references"]
        for ref in record["references"]:
            assert ref["ref_id"] in registry_ids
            metadata = ref["match_metadata"]
            assert metadata["match_type"] == "manual"
            assert metadata["confidence"] == "high"
            assert metadata["notes"]


def test_pubrev077_review_bundle_is_mergeable_without_unresolved_atoms() -> None:
    summary = merge_audit_manifest_with_review_bundles(
        manifest_path=MANIFEST_PATH,
        review_bundle_paths=[BUNDLE_PATH],
        dry_run=True,
    )

    assert summary["bundle_entry_count"] == len(PROMOTED_ATOMS)
    assert summary["skipped_unresolved_atom_count"] == 0
