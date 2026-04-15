from __future__ import annotations

import json
from pathlib import Path

from sciona.atoms.supabase_backfill import (
    build_atom_reference_row,
    build_evidence_rows,
    build_io_spec_rows,
    build_parameter_rows,
    build_ref_key,
    build_registry_row,
    build_rollup_row,
    build_uncertainty_rows,
    build_verification_match_row,
    choose_technical_content,
    dedupe_audit_rollup_rows,
    dedupe_technical_description_rows,
    dedupe_uncertainty_rows,
    dedupe_verification_match_rows,
    derive_atom_fqdn,
    extract_fqdn,
    input_name_mismatch,
    iter_reference_files,
    load_registry,
    map_source,
    namespace_from_path,
    normalize_acceptability_band,
    normalize_verification_level,
)


def test_derive_atom_fqdn_supports_namespace_package_roots() -> None:
    atoms_root = Path("/tmp/provider/src/sciona/atoms")
    cdg_path = atoms_root / "signal_processing" / "biosppy" / "cdg.json"
    assert (
        derive_atom_fqdn(cdg_path, atoms_root, "online_filter")
        == "sciona.atoms.signal_processing.biosppy.online_filter"
    )


def test_namespace_from_path_handles_namespace_and_artifacts() -> None:
    assert (
        namespace_from_path(Path("/tmp/repo/src/sciona/atoms/signal_processing/biosppy/matches.json"))
        == "sciona.atoms.signal_processing.biosppy"
    )
    assert (
        namespace_from_path(Path("ageoa/mint/_artifacts/apc_module/uncertainty.json"))
        == "ageoa.mint"
    )


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
    assert extract_fqdn("ageoa.algorithms.graph.bellman_ford@ageoa/algorithms/graph.py:174") == (
        "ageoa.algorithms.graph.bellman_ford"
    )
    assert build_ref_key("almgren2000", {"doi": "10.1000/example"}) == "10.1000/example"
    assert map_source({"match_type": "ast_subgraph"}) == "llm_extracted"
    row = build_registry_row("clrs2009", {"type": "book", "title": "CLRS"})
    assert row["bibtex_key"] == "clrs2009"
    atom_ref = build_atom_reference_row(
        "atom-1",
        "clrs2009",
        {"title": "Introduction to Algorithms", "authors": ["Cormen"], "year": 2009, "url": "https://example.test"},
        {"notes": "core citation", "confidence": "high", "matched_nodes": ["shortest_path"], "match_type": "manual"},
    )
    assert atom_ref["source"] == "manual"
    assert atom_ref["matched_nodes"] == ["shortest_path"]


def test_load_registry_accepts_wrapped_and_plain_payloads(tmp_path: Path) -> None:
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"references": {"ref_a": {"title": "A"}}}))
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps({"ref_b": {"title": "B"}}))
    assert load_registry(wrapped) == {"ref_a": {"title": "A"}}
    assert load_registry(plain) == {"ref_b": {"title": "B"}}


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
        [{"scalar_factor": 0.7, "confidence": 0.9, "input_regime": "shape=(256,)"}],
    )
    assert rows[0]["mode"] == "empirical"
    assert rows[0]["input_regime"] == "shape=(256,)"


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
