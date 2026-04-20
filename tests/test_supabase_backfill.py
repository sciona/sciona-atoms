from __future__ import annotations

import json
import os
from pathlib import Path

from sciona.atoms.provider_inventory import discover_audit_manifest_path, discover_audit_manifest_paths
from sciona.atoms.supabase_backfill import (
    build_atom_reference_row,
    build_dejargonized_description_row,
    build_evidence_rows,
    build_io_spec_rows,
    build_manifest_io_spec_rows,
    build_manifest_reference_binding,
    build_parameter_rows,
    build_ref_key,
    build_registry_row,
    build_rollup_row,
    build_uncertainty_rows,
    build_verification_match_row,
    choose_dejargonized_content,
    choose_technical_content,
    dedupe_audit_rollup_rows,
    dedupe_technical_description_rows,
    dedupe_uncertainty_rows,
    dedupe_verification_match_rows,
    derive_atom_fqdn,
    extract_fqdn,
    input_name_mismatch,
    iter_reference_files,
    load_manifest_entries,
    load_registry,
    map_source,
    namespace_from_path,
    normalize_acceptability_band,
    normalize_reference_type,
    normalize_uncertainty_mode,
    normalize_verification_level,
)


ROOT = Path(__file__).resolve().parents[1]
_LEGACY_PROVIDER_LABEL = "ageo" + "-atoms"
_LEGACY_NAMESPACE_LABEL = "age" + "oa"


def test_derive_atom_fqdn_supports_namespace_package_roots() -> None:
    atoms_root = Path("/tmp/provider/src/sciona/atoms")
    cdg_path = atoms_root / "signal_processing" / "biosppy" / "cdg.json"
    assert (
        derive_atom_fqdn(cdg_path, atoms_root, "online_filter")
        == "sciona.atoms.signal_processing.biosppy.online_filter"
    )


def test_namespace_from_path_handles_namespace_and_artifact_boundaries() -> None:
    assert (
        namespace_from_path(Path("/tmp/repo/src/sciona/atoms/signal_processing/biosppy/matches.json"))
        == "sciona.atoms.signal_processing.biosppy"
    )
    assert (
        namespace_from_path(Path("/tmp/repo/src/sciona/atoms/bio/mint/_artifacts/apc_module/uncertainty.json"))
        == "sciona.atoms.bio.mint"
    )


def test_discover_audit_manifest_path_uses_provider_owned_manifest(tmp_path: Path) -> None:
    workspace = tmp_path / "audit-manifest-workspace"
    manifest_path = workspace / "sciona-atoms" / "data" / "audit_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"schema_version": "1.1", "atoms": [{"atom_name": "sciona.atoms.demo.scale"}]})
    )

    resolved = discover_audit_manifest_path(base_dir=workspace)
    assert resolved == manifest_path.resolve()

    text = resolved.read_text()
    assert _LEGACY_PROVIDER_LABEL not in text
    assert _LEGACY_NAMESPACE_LABEL not in text
    assert "sciona.atoms.demo.scale" in text


def test_load_manifest_entries_merges_provider_owned_manifests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "audit-manifest-workspace"
    base_manifest_path = workspace / "sciona-atoms" / "data" / "audit_manifest.json"
    bio_manifest_path = workspace / "sciona-atoms-bio" / "data" / "audit_manifest.json"
    physics_manifest_path = workspace / "sciona-atoms-physics" / "data" / "audit_manifest.json"
    base_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    bio_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    physics_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    (workspace / "sciona-atoms-bio" / "src" / "sciona" / "atoms" / "bio").mkdir(parents=True)
    (workspace / "sciona-atoms-physics" / "src" / "sciona" / "atoms" / "physics").mkdir(parents=True)
    base_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "atoms": [
                    {
                        "atom_name": "sciona.atoms.bio.demo.scale",
                        "review_status": "reviewed_pending",
                    },
                    {
                        "atom_name": "sciona.atoms.demo.identity",
                        "review_status": "reviewed",
                    },
                ],
            }
        )
    )
    bio_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "atoms": [
                    {
                        "atom_name": "sciona.atoms.bio.demo.scale",
                        "review_status": "reviewed",
                    },
                    {
                        "atom_name": "sciona.atoms.bio.demo.offset",
                        "review_status": "reviewed",
                    },
                ],
            }
        )
    )
    physics_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "atoms": [
                    {
                        "atom_name": "sciona.atoms.bio.demo.scale",
                        "review_status": "stale_cross_repo_copy",
                    },
                    {
                        "atom_name": "sciona.atoms.physics.demo.energy",
                        "review_status": "reviewed",
                    },
                ],
            }
        )
    )

    monkeypatch.setenv("SCIONA_PROVIDER_WORKSPACE_ROOT", str(workspace))

    assert discover_audit_manifest_paths() == (
        base_manifest_path.resolve(),
        bio_manifest_path.resolve(),
        physics_manifest_path.resolve(),
    )
    entries = load_manifest_entries()
    by_name = {entry["atom_name"]: entry for entry in entries}
    assert by_name["sciona.atoms.bio.demo.scale"]["review_status"] == "reviewed"
    assert by_name["sciona.atoms.demo.identity"]["review_status"] == "reviewed"
    assert by_name["sciona.atoms.bio.demo.offset"]["review_status"] == "reviewed"
    assert by_name["sciona.atoms.physics.demo.energy"]["review_status"] == "reviewed"


def test_repository_audit_manifest_has_no_legacy_references() -> None:
    text = (ROOT / "data" / "audit_manifest.json").read_text()
    assert _LEGACY_PROVIDER_LABEL not in text
    assert _LEGACY_NAMESPACE_LABEL not in text
    assert "sciona.atoms.signal_processing.biosppy.ecg_zz2018_d12.computekurtosissqi" in text


def test_build_io_spec_rows_maps_inputs_and_outputs() -> None:
    rows = build_io_spec_rows(
        "atom-1",
        {
            "inputs": [{"name": "x", "type_desc": "float", "constraints": ">= 0"}],
            "outputs": [{"name": "y"}],
        },
    )
    assert rows[0]["direction"] == "input"
    assert rows[1]["direction"] == "output"


def test_build_manifest_io_spec_rows_derives_inputs_and_default_output() -> None:
    rows = build_manifest_io_spec_rows(
        "atom-1",
        {
            "argument_details": [
                {"name": "x", "annotation": "ArrayLike", "required": True},
                {"name": "dtype", "annotation": "DTypeLike", "required": False},
            ],
            "return_annotation": "np.ndarray",
        },
    )

    assert [row["direction"] for row in rows] == ["input", "input", "output"]
    assert rows[0]["name"] == "x"
    assert rows[0]["type_desc"] == "ArrayLike"
    assert rows[1]["required"] is False
    assert rows[2]["name"] == "result"
    assert rows[2]["type_desc"] == "np.ndarray"


def test_build_manifest_io_spec_rows_supports_zero_input_atoms() -> None:
    rows = build_manifest_io_spec_rows(
        "atom-1",
        {
            "argument_details": [],
            "return_annotation": "YawLockState",
        },
    )

    assert rows == [
        {
            "atom_id": "atom-1",
            "version_id": None,
            "direction": "output",
            "name": "result",
            "type_desc": "YawLockState",
            "constraints": "",
            "required": True,
            "default_value_repr": "",
            "ordinal": 0,
        }
    ]


def test_input_name_mismatch_only_warns_when_manifest_present() -> None:
    assert not input_name_mismatch(["x"], [])
    assert input_name_mismatch(["x"], ["y"])


def test_build_parameter_rows_appends_varargs_and_kwargs() -> None:
    rows = build_parameter_rows(
        "atom-1",
        {
            "argument_details": [{"name": "q", "annotation": "np.ndarray", "required": True}],
            "uses_varargs": True,
            "uses_kwargs": True,
        },
    )
    assert [row["name"] for row in rows] == ["q", "*args", "**kwargs"]


def test_choose_technical_content_prefers_docstring_summary() -> None:
    assert (
        choose_technical_content({"docstring_summary": "Doc summary"}, {"description": "Fallback"})
        == "Doc summary"
    )


def test_choose_dejargonized_content_normalizes_and_terminates_sentence() -> None:
    assert (
        choose_dejargonized_content(
            {"docstring_summary": "Estimate_event_rate from signal"},
            {"description": "Fallback"},
        )
        == "Estimate event rate from signal."
    )


def test_build_dejargonized_description_row_uses_plain_language_kind() -> None:
    row = build_dejargonized_description_row("atom-1", "Estimate event rate from signal.")
    assert row["kind"] == "dejargonized"
    assert row["language"] == "en"
    assert row["jargon_score"] < 0.4


def test_dedupe_technical_description_rows_collapses_conflicting_keys() -> None:
    rows = [
        {
            "atom_id": "atom-1",
            "kind": "technical",
            "language": "en",
            "content": "First",
        },
        {
            "atom_id": "atom-1",
            "kind": "technical",
            "language": "en",
            "content": "Second",
        },
        {
            "atom_id": "atom-2",
            "kind": "technical",
            "language": "en",
            "content": "Third",
        },
    ]

    deduped = dedupe_technical_description_rows(rows)

    assert deduped == [
        {
            "atom_id": "atom-1",
            "kind": "technical",
            "language": "en",
            "content": "Second",
        },
        {
            "atom_id": "atom-2",
            "kind": "technical",
            "language": "en",
            "content": "Third",
        },
    ]


def test_dedupe_technical_description_rows_prefers_richer_content_deterministically() -> None:
    rows = [
        {
            "atom_id": "atom-1",
            "kind": "technical",
            "language": "en",
            "content": "Brief",
        },
        {
            "atom_id": "atom-1",
            "kind": "technical",
            "language": "en",
            "content": "A much longer technical description",
        },
    ]

    assert dedupe_technical_description_rows(list(reversed(rows))) == [rows[1]]


def test_registry_and_reference_helpers_roundtrip() -> None:
    assert extract_fqdn("sciona.atoms.algorithms.graph.bellman_ford@sciona/atoms/algorithms/graph.py:174") == (
        "sciona.atoms.algorithms.graph.bellman_ford"
    )
    assert build_ref_key("almgren2000", {"doi": "10.1000/example"}) == "10.1000/example"
    assert map_source({"match_type": "ast_subgraph"}) == "llm_extracted"
    row = build_registry_row("clrs2009", {"type": "book", "title": "CLRS"})
    assert row["bibtex_key"] == "clrs2009"
    assert normalize_reference_type("software") == "repository"
    assert normalize_reference_type("technical-report") == "paper"
    assert build_registry_row("repo_scipy", {"type": "software"})["ref_type"] == "repository"
    atom_ref = build_atom_reference_row(
        "atom-1",
        "clrs2009",
        {"title": "Introduction to Algorithms", "authors": ["Cormen"], "year": 2009, "url": "https://example.test"},
        {"notes": "core citation", "confidence": "high", "matched_nodes": ["shortest_path"], "match_type": "manual"},
    )
    assert atom_ref["source"] == "manual"
    assert atom_ref["matched_nodes"] == ["shortest_path"]


def test_build_manifest_reference_binding_derives_upstream_docs_reference() -> None:
    binding = build_manifest_reference_binding(
        {
            "atom_name": "sciona.atoms.numpy.arrays.dot",
            "has_references": True,
            "references_status": "pass",
            "upstream_symbols": {
                "module": "numpy",
                "function": "dot",
                "notes": "Uses the installed NumPy package as the upstream signature source.",
            },
        }
    )

    assert binding is not None
    ref_id, registry_entry, match_metadata = binding
    assert ref_id == "upstream:numpy.dot"
    assert registry_entry["title"] == "API reference for numpy.dot"
    assert registry_entry["url"] == "https://numpy.org/doc/stable/reference/generated/numpy.dot.html"
    assert match_metadata["confidence"] == "medium"


def test_build_manifest_reference_binding_returns_none_without_upstream_module() -> None:
    assert build_manifest_reference_binding({"atom_name": "sciona.atoms.algorithms.graph.bfs"}) is None


def test_load_registry_accepts_wrapped_and_plain_payloads(tmp_path: Path) -> None:
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"references": {"ref_a": {"title": "A"}}}))
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps({"ref_b": {"title": "B"}}))
    assert load_registry(wrapped) == {"ref_a": {"title": "A"}}
    assert load_registry(plain) == {"ref_b": {"title": "B"}}


def test_load_registry_merges_provider_registries(monkeypatch, tmp_path: Path) -> None:
    repo_a = tmp_path / "sciona-atoms"
    repo_b = tmp_path / "sciona-atoms-signal"
    (repo_a / "data" / "references").mkdir(parents=True)
    (repo_b / "data" / "references").mkdir(parents=True)
    (repo_a / "data" / "references" / "registry.json").write_text(
        json.dumps({"references": {"ref_a": {"title": "A"}}})
    )
    (repo_b / "data" / "references" / "registry.json").write_text(
        json.dumps({"references": {"ref_b": {"title": "B"}}})
    )
    monkeypatch.setenv("SCIONA_ATOM_PROVIDER_ROOTS", os.pathsep.join([str(repo_a), str(repo_b)]))
    assert load_registry() == {"ref_a": {"title": "A"}, "ref_b": {"title": "B"}}


def test_load_registry_rejects_conflicting_provider_entries(monkeypatch, tmp_path: Path) -> None:
    repo_a = tmp_path / "sciona-atoms"
    repo_b = tmp_path / "sciona-atoms-signal"
    (repo_a / "data" / "references").mkdir(parents=True)
    (repo_b / "data" / "references").mkdir(parents=True)
    (repo_a / "data" / "references" / "registry.json").write_text(
        json.dumps({"references": {"shared": {"title": "A"}}})
    )
    (repo_b / "data" / "references" / "registry.json").write_text(
        json.dumps({"references": {"shared": {"title": "B"}}})
    )
    monkeypatch.setenv("SCIONA_ATOM_PROVIDER_ROOTS", os.pathsep.join([str(repo_a), str(repo_b)]))
    try:
        load_registry()
    except ValueError as exc:
        assert "Conflicting registry entry" in str(exc)
    else:
        raise AssertionError("Expected conflicting registry entries to fail")


def test_load_registry_uses_provider_owned_registry_without_ageo_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "sciona-atoms"
    (repo / "data" / "references").mkdir(parents=True)
    (repo / "data" / "references" / "registry.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "references": {
                    "signal2026": {
                        "ref_id": "signal2026",
                        "type": "paper",
                        "title": "Canonical Signal Reference",
                    }
                },
            }
        )
    )
    monkeypatch.setenv("SCIONA_ATOM_PROVIDER_ROOTS", str(repo))
    assert load_registry() == {
        "signal2026": {
            "ref_id": "signal2026",
            "type": "paper",
            "title": "Canonical Signal Reference",
        }
    }


def test_iter_reference_files_accepts_multiple_roots(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    (left / "alpha").mkdir(parents=True)
    (right / "beta").mkdir(parents=True)
    (left / "alpha" / "references.json").write_text("{}")
    (right / "beta" / "references.json").write_text("{}")
    files = iter_reference_files((left, right))
    assert files == [
        (left / "alpha" / "references.json").resolve(),
        (right / "beta" / "references.json").resolve(),
    ]


def test_build_rollup_and_evidence_rows_cover_expected_fields() -> None:
    entry = {
        "overall_verdict": "pass",
        "structural_status": "pass",
        "semantic_status": "pass",
        "risk_tier": "low",
        "runtime_status": "pass",
        "parity_coverage_level": "positive_and_negative",
    }
    rollup = build_rollup_row("atom-1", entry)
    evidence = build_evidence_rows("atom-1", entry)
    assert rollup["overall_verdict"] == "pass"
    assert {row["audit_type"] for row in evidence} >= {
        "structural_audit",
        "semantic_audit",
        "risk_assessment",
        "parity_check",
        "smoke_test",
    }


def test_normalize_acceptability_band_collapses_newer_manifest_values() -> None:
    assert normalize_acceptability_band("review_ready", acceptability_score=91) == (
        "acceptable_with_limits"
    )
    assert normalize_acceptability_band("misleading_candidate", acceptability_score=42) == "unknown"
    assert normalize_acceptability_band("broken_candidate", acceptability_score=10) == "unknown"
    assert normalize_acceptability_band("acceptable_with_limits_candidate", acceptability_score=75) == (
        "acceptable_with_limits_candidate"
    )
    assert normalize_acceptability_band("limited_acceptability", acceptability_score=60) == (
        "limited_acceptability"
    )


def test_dedupe_audit_rollup_rows_collapses_duplicate_atom_ids() -> None:
    rows = [
        {
            "atom_id": "atom-1",
            "overall_verdict": "misleading",
            "acceptability_score": 49,
            "risk_score": 27,
            "parity_case_count": 0,
            "parity_fixture_count": 0,
        },
        {
            "atom_id": "atom-1",
            "overall_verdict": "acceptable_with_limits",
            "acceptability_score": 91,
            "risk_score": 15,
            "parity_case_count": 0,
            "parity_fixture_count": 0,
        },
        {"atom_id": "atom-2", "overall_verdict": "broken"},
    ]

    deduped = dedupe_audit_rollup_rows(list(reversed(rows)))

    assert deduped == [
        {
            "atom_id": "atom-1",
            "overall_verdict": "acceptable_with_limits",
            "acceptability_score": 91,
            "risk_score": 15,
            "parity_case_count": 0,
            "parity_fixture_count": 0,
        },
        {"atom_id": "atom-2", "overall_verdict": "broken"},
    ]


def test_dedupe_uncertainty_rows_collapses_exact_duplicates() -> None:
    rows = [
        {"atom_id": "a", "version_id": None, "mode": "empirical", "scalar_factor": 0.7},
        {"atom_id": "a", "version_id": None, "mode": "empirical", "scalar_factor": 0.7},
        {"atom_id": "a", "version_id": None, "mode": "empirical", "scalar_factor": 0.8},
    ]

    assert dedupe_uncertainty_rows(list(reversed(rows))) == [
        {"atom_id": "a", "version_id": None, "mode": "empirical", "scalar_factor": 0.7},
        {"atom_id": "a", "version_id": None, "mode": "empirical", "scalar_factor": 0.8},
    ]


def test_dedupe_verification_match_rows_collapses_exact_duplicates() -> None:
    rows = [
        {"atom_id": "a", "version_id": None, "predicate_id": "p", "candidate_name": "c"},
        {"atom_id": "a", "version_id": None, "predicate_id": "p", "candidate_name": "c"},
        {"atom_id": "a", "version_id": None, "predicate_id": "q", "candidate_name": "c"},
    ]

    assert dedupe_verification_match_rows(list(reversed(rows))) == [
        {"atom_id": "a", "version_id": None, "predicate_id": "p", "candidate_name": "c"},
        {"atom_id": "a", "version_id": None, "predicate_id": "q", "candidate_name": "c"},
    ]


def test_build_uncertainty_rows_maps_optional_fields() -> None:
    rows = build_uncertainty_rows(
        "atom-1",
        [
            {"scalar_factor": 0.7, "confidence": 0.9, "input_regime": "shape=(256,)"},
            {"mode": "source_review", "scalar_factor": 0.8, "confidence": 0.85},
        ],
    )
    assert rows[0]["mode"] == "empirical"
    assert rows[0]["input_regime"] == "shape=(256,)"
    assert normalize_uncertainty_mode("source-review") == "empirical"
    assert rows[1]["mode"] == "empirical"


def test_verification_helpers_guard_unknown_levels() -> None:
    assert normalize_verification_level("surprising") == "unverified"
    row = build_verification_match_row(
        "atom-2",
        {
            "pdg_node": {"predicate_id": "isleapyear", "statement": "(year: Any) -> Any", "informal_desc": ""},
            "all_candidates": [{"name": "candidate"}],
            "all_verifications": [{"status": "attempted"}],
        },
    )
    assert row["predicate_id"] == "isleapyear"
    assert row["verification_level"] == "unverified"



class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, pages):
        self._pages = pages
        self._range = None

    def select(self, _columns):
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    def execute(self):
        start = 0 if self._range is None else self._range[0]
        page_size = len(self._pages[0]) if self._pages else 1000
        page_index = start // page_size if page_size else 0
        return _FakeResponse(self._pages[page_index] if page_index < len(self._pages) else [])


class _FakeClient:
    def __init__(self, pages):
        self._pages = pages

    def table(self, name):
        assert name == "atoms"
        return _FakeQuery(self._pages)


def test_fetch_atom_lookup_paginates() -> None:
    from sciona.atoms.supabase_backfill import fetch_atom_lookup

    pages = [
        [{"fqdn": f"atom.{index}", "atom_id": f"id-{index}"} for index in range(1000)],
        [{"fqdn": "atom.1000", "atom_id": "id-1000"}],
    ]
    client = _FakeClient(pages)
    rows = fetch_atom_lookup(client)
    assert len(rows) == 1001
    assert rows["atom.1000"] == "id-1000"
