from __future__ import annotations

import json
import importlib
import sys
import textwrap
from pathlib import Path

from sciona.atoms.audit_review_bundles import (
    discover_review_bundle_paths,
    load_review_bundle_entries,
    merge_audit_manifest_with_review_bundles,
)


def test_normalized_legacy_row_bundle_is_not_shadowed_by_empty_atoms() -> None:
    """Rows generated from a compact legacy bundle must reach the merger."""
    root = Path(__file__).resolve().parents[1]
    entries = load_review_bundle_entries(
        root / "data/review_bundles/adaptive_gauss_kronrod_quadrature.review_bundle.json"
    )
    assert entries


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


def test_workspace_callable_replaces_stale_cached_provider_module(tmp_path: Path) -> None:
    installed = tmp_path / "installed"
    workspace = tmp_path / "workspace"
    package_rel = Path("demo_provider") / "family"
    _write(installed / package_rel / "__init__.py", "from .atoms import *\n")
    _write(installed / package_rel / "atoms.py", "def old_atom(value):\n    return value\n")
    _write(workspace / "src" / package_rel / "__init__.py", "from .atoms import *\n")
    _write(
        workspace / "src" / package_rel / "atoms.py",
        "def new_atom(value, scale=1):\n    return value * scale\n",
    )
    sys.path.insert(0, str(installed))
    try:
        importlib.import_module("demo_provider.family")
        bundle_path = workspace / "data" / "audit_reviews" / "provider.json"
        _write(
            bundle_path,
            """
            {
              "schema_version": "1.0",
              "provider_repo": "demo-provider",
              "rows": [{
                "atom_name": "demo_provider.family.new_atom",
                "review_status": "pending",
                "trust_readiness": "unreviewed"
              }]
            }
            """,
        )
        entries = load_review_bundle_entries(bundle_path)
        # Exercise the lower-level merger with the workspace-owned bundle.
        from sciona.atoms.audit_review_bundles import merge_audit_manifest_entries

        rows, skipped = merge_audit_manifest_entries([], entries)
        assert skipped == []
        assert rows[0]["atom_name"] == "demo_provider.family.new_atom"
        assert rows[0]["argument_names"] == ["value", "scale"]
    finally:
        sys.path.remove(str(installed))
        sys.modules.pop("demo_provider.family.atoms", None)
        sys.modules.pop("demo_provider.family", None)
        sys.modules.pop("demo_provider", None)


def test_discover_review_bundle_paths_is_provider_owned_and_sorted(tmp_path: Path) -> None:
    workspace = tmp_path
    _write(
        workspace / "sciona-atoms" / "data" / "audit_reviews" / "b.json",
        """
        {"schema_version": "1.0", "atoms": []}
        """,
    )
    _write(
        workspace / "sciona-atoms-signal" / "data" / "audit_reviews" / "a.json",
        """
        {"schema_version": "1.0", "atoms": []}
        """,
    )
    _write(
        workspace / "sciona-atoms-bio" / "src" / "sciona" / "atoms" / "bio" / "family" / "review_bundle.json",
        """
        {"schema_version": "1.0", "rows": []}
        """,
    )

    paths = discover_review_bundle_paths(base_dir=workspace)

    assert [path.name for path in paths] == ["b.json", "review_bundle.json", "a.json"]
    assert "sciona-atoms" in str(paths[0])
    assert "sciona-atoms-bio" in str(paths[1])
    assert "sciona-atoms-signal" in str(paths[2])


def test_merge_audit_manifest_with_review_bundles_promotes_and_creates_entries(tmp_path: Path) -> None:
    workspace = tmp_path
    manifest_path = workspace / "sciona-atoms" / "data" / "audit_manifest.json"
    review_bundle_path = workspace / "sciona-atoms" / "data" / "audit_reviews" / "provider.json"
    review_bundle_two_path = workspace / "sciona-atoms-signal" / "data" / "audit_reviews" / "signal.json"
    package_root = workspace / "pkgs"
    package_root.mkdir(parents=True, exist_ok=True)
    _write(
        package_root / "demoatoms.py",
        """
        def new_contract(value: int, scale: float = 1.0) -> float:
            \"\"\"Scale a value deterministically.\"\"\"
            return value * scale

        def signal_contract(signal, fs: int):
            \"\"\"Measure a signal property.\"\"\"
            return signal
        """,
    )
    sys.path.insert(0, str(package_root))

    _write(
        manifest_path,
        """
        {
          "schema_version": "1.1",
          "metadata": {
            "generated_at": "2026-04-15T00:00:00Z",
            "generator": "existing-generator"
          },
          "atoms": [
                {
                  "atom_name": "sciona.atoms.demo.existing",
                  "atom_key": "demo/existing",
                  "review_status": "draft",
                  "review_priority": "review_later",
              "structural_status": "unknown",
              "semantic_status": "unknown",
              "runtime_status": "unknown",
              "developer_semantics_status": "unknown",
              "review_record_path": "legacy/manual.md",
              "source_kind": "hand_written"
            }
          ]
        }
        """,
    )
    _write(
        review_bundle_path,
        """
        {
          "schema_version": "1.0",
          "provider_repo": "sciona-atoms",
          "rows": [
            {
                  "atom_key": "demo/existing",
                  "trust_readiness": "ready_for_manifest_merge",
                  "semantic_verdict": "supported",
                  "developer_semantic_verdict": "aligned_to_registered_atoms"
                },
                {
                  "atom_fqdn": "demoatoms.new_contract@src/demoatoms.py:1",
                  "module_path": "src/demoatoms.py",
                  "upstream_symbols": {
                    "module": "demo",
                    "function": "new_contract"
                  },
                  "upstream_version": "1.0",
                  "has_references": true,
                  "references_status": "pass",
                  "trust_readiness": "ready_for_manifest_merge",
                  "semantic_verdict": "supported",
                  "developer_semantic_verdict": "aligned_to_registered_atoms"
                }
          ]
        }
        """,
    )
    _write(
        review_bundle_two_path,
        """
        {
          "schema_version": "1.0",
          "provider_repo": "sciona-atoms-signal",
          "rows": [
            {
              "atom_name": "demoatoms.signal_contract",
              "trust_readiness": "needs_followup",
              "semantic_verdict": "supported",
              "developer_semantic_verdict": "aligned_to_registered_atoms",
              "required_actions": ["add stronger runtime evidence"],
              "review_record_path": "signals/review.md"
            }
          ]
        }
        """,
    )

    summary = merge_audit_manifest_with_review_bundles(
        manifest_path=manifest_path,
        base_dir=workspace,
    )
    merged = json.loads(manifest_path.read_text(encoding="utf-8"))
    atoms = merged["atoms"]

    assert summary["bundle_entry_count"] == 3
    assert summary["created_entry_count"] == 2
    assert summary["updated_entry_count"] == 1
    assert {entry["atom_name"] for entry in atoms} == {
        "sciona.atoms.demo.existing",
        "demoatoms.new_contract",
        "demoatoms.signal_contract",
    }
    by_name = {entry["atom_name"]: entry for entry in atoms}
    existing = by_name["sciona.atoms.demo.existing"]
    created = by_name["demoatoms.new_contract"]
    signal = by_name["demoatoms.signal_contract"]
    assert existing["review_status"] == "approved"
    assert existing["atom_key"] == "demo/existing"
    assert existing["review_priority"] == "review_now"
    assert existing["structural_status"] == "pass"
    assert existing["semantic_status"] == "pass"
    assert existing["runtime_status"] == "pass"
    assert existing["review_record_path"] == "data/audit_reviews/provider.json"
    assert created["atom_name"] == "demoatoms.new_contract"
    assert created["review_status"] == "approved"
    assert created["runtime_status"] == "pass"
    assert created["argument_names"] == ["value", "scale"]
    assert created["return_annotation"] == "float"
    assert created["docstring_summary"] == "Scale a value deterministically."
    assert created["review_record_path"] == "data/audit_reviews/provider.json"
    assert created["module_path"] == "src/demoatoms.py"
    assert created["upstream_symbols"] == {"module": "demo", "function": "new_contract"}
    assert created["upstream_version"] == "1.0"
    assert created["has_references"] is True
    assert created["references_status"] == "pass"
    assert signal["review_record_path"] == "signals/review.md"
    assert signal["review_status"] == "missing"
    assert signal["review_required_actions"] == ["add stronger runtime evidence"]
    sys.path.remove(str(package_root))


def test_merge_refreshes_existing_structural_fields_from_live_callable(tmp_path: Path) -> None:
    workspace = tmp_path
    manifest_path = workspace / "sciona-atoms" / "data" / "audit_manifest.json"
    review_bundle_path = workspace / "sciona-atoms-fintech" / "data" / "audit_reviews" / "provider.json"
    package_root = workspace / "pkgs"
    package_root.mkdir(parents=True, exist_ok=True)
    _write(
        package_root / "refreshatoms.py",
        """
        def initialize_state(_trigger: None = None) -> tuple[float, float]:
            \"\"\"Initialize state using an optional sentinel input.\"\"\"
            return (0.0, 0.0)
        """,
    )
    sys.path.insert(0, str(package_root))

    _write(
        manifest_path,
        """
        {
          "schema_version": "1.1",
          "metadata": {},
          "atoms": [
            {
              "atom_name": "refreshatoms.initialize_state",
              "atom_key": "refreshatoms.initialize_state",
              "module_import_path": "refreshatoms",
              "module_path": "/tmp/icontract/_checkers.py",
              "wrapper_symbol": "initialize_state",
              "argument_names": [],
              "argument_details": [],
              "return_annotation": "tuple[float, float]",
              "docstring_summary": "stale summary",
              "has_docstring": false,
              "review_status": "draft",
              "review_priority": "review_later"
            }
          ]
        }
        """,
    )
    _write(
        review_bundle_path,
        """
        {
          "schema_version": "1.0",
          "provider_repo": "sciona-atoms-fintech",
          "rows": [
            {
              "atom_name": "refreshatoms.initialize_state",
              "trust_readiness": "ready_for_manifest_merge",
              "semantic_verdict": "supported",
              "developer_semantic_verdict": "aligned_to_registered_atoms"
            }
          ]
        }
        """,
    )

    merge_audit_manifest_with_review_bundles(
        manifest_path=manifest_path,
        base_dir=workspace,
    )
    merged = json.loads(manifest_path.read_text(encoding="utf-8"))
    atom = merged["atoms"][0]

    assert atom["argument_names"] == ["_trigger"]
    assert atom["argument_details"] == [
        {
            "name": "_trigger",
            "annotation": "None",
            "required": False,
            "kind": "positional_or_keyword",
        }
    ]
    assert atom["docstring_summary"] == "Initialize state using an optional sentinel input."
    assert atom["module_path"].endswith("refreshatoms.py")
    sys.path.remove(str(package_root))


def test_merge_preserves_explicit_empty_row_lists_over_bundle_level_blockers(tmp_path: Path) -> None:
    workspace = tmp_path
    manifest_path = workspace / "sciona-atoms" / "data" / "audit_manifest.json"
    review_bundle_path = (
        workspace
        / "sciona-atoms-bio"
        / "src"
        / "sciona"
        / "atoms"
        / "bio"
        / "family"
        / "review_bundle.json"
    )
    package_root = workspace / "pkgs"
    package_root.mkdir(parents=True, exist_ok=True)
    _write(
        package_root / "partialbundleatoms.py",
        """
        def promoted_atom(value: int) -> int:
            \"\"\"Return the same value.\"\"\"
            return value
        """,
    )
    sys.path.insert(0, str(package_root))

    _write(
        manifest_path,
        """
        {
          "schema_version": "1.1",
          "metadata": {},
          "atoms": []
        }
        """,
    )
    _write(
        review_bundle_path,
        """
        {
          "schema_version": "1.0",
          "provider_repo": "sciona-atoms-bio",
          "review_status": "partial",
          "trust_readiness": "blocked_on_uncertainty_backfill",
          "limitations": ["family-level blocker"],
          "required_actions": ["family-level action"],
          "rows": [
            {
              "atom_name": "partialbundleatoms.promoted_atom",
              "trust_readiness": "ready_for_manifest_merge",
              "semantic_verdict": "supported",
              "developer_semantic_verdict": "aligned_to_registered_atoms",
              "limitations": [],
              "required_actions": [],
              "authoritative_sources": []
            }
          ]
        }
        """,
    )

    merge_audit_manifest_with_review_bundles(
        manifest_path=manifest_path,
        base_dir=workspace,
    )
    merged = json.loads(manifest_path.read_text(encoding="utf-8"))
    atom = merged["atoms"][0]

    assert atom["atom_name"] == "partialbundleatoms.promoted_atom"
    assert atom["review_status"] == "approved"
    assert atom["review_required_actions"] == []
    assert atom["review_limitations"] == []
    assert atom["trust_blockers"] == []
    assert atom["authoritative_sources"] == []
    sys.path.remove(str(package_root))


def test_merge_creates_entries_from_provider_src_roots(tmp_path: Path) -> None:
    workspace = tmp_path
    manifest_path = workspace / "sciona-atoms" / "data" / "audit_manifest.json"
    review_bundle_path = workspace / "sciona-atoms-signal" / "data" / "review_bundles" / "provider.json"
    provider_module = (
        workspace
        / "sciona-atoms-signal"
        / "src"
        / "sciona"
        / "atoms"
        / "signal_processing"
        / "demo_family"
        / "atoms.py"
    )

    _write(
        provider_module,
        """
        def detect_demo_events(signal: list[float], threshold: float = 0.5) -> list[float]:
            \"\"\"Return the input signal for deterministic testing.\"\"\"
            return signal
        """,
    )
    _write(
        provider_module.parent / "__init__.py",
        """
        from .atoms import detect_demo_events
        """,
    )
    _write(workspace / "sciona-atoms-signal" / "src" / "sciona" / "__init__.py", "")
    _write(workspace / "sciona-atoms-signal" / "src" / "sciona" / "atoms" / "__init__.py", "")
    _write(
        workspace / "sciona-atoms-signal" / "src" / "sciona" / "atoms" / "signal_processing" / "__init__.py",
        "",
    )
    _write(
        manifest_path,
        """
        {
          "schema_version": "1.1",
          "metadata": {},
          "atoms": []
        }
        """,
    )
    _write(
        review_bundle_path,
        """
        {
          "schema_version": "1.0",
          "provider_repo": "sciona-atoms-signal",
          "rows": [
            {
              "atom_name": "sciona.atoms.signal_processing.demo_family.detect_demo_events",
              "trust_readiness": "ready_for_manifest_merge",
              "semantic_verdict": "supported",
              "developer_semantic_verdict": "aligned_to_registered_atoms",
              "required_actions": [],
              "limitations": []
            }
          ]
        }
        """,
    )

    merge_audit_manifest_with_review_bundles(
        manifest_path=manifest_path,
        base_dir=workspace,
    )
    merged = json.loads(manifest_path.read_text(encoding="utf-8"))
    atom = merged["atoms"][0]

    assert atom["atom_name"] == "sciona.atoms.signal_processing.demo_family.detect_demo_events"
    assert atom["review_status"] == "approved"
    assert atom["module_path"].endswith("demo_family/atoms.py")
    assert atom["argument_names"] == ["signal", "threshold"]

    for module_name in (
        "sciona.atoms.signal_processing.demo_family.atoms",
        "sciona.atoms.signal_processing.demo_family",
        "sciona.atoms.signal_processing",
    ):
        sys.modules.pop(module_name, None)
    atoms_package = sys.modules.get("sciona.atoms")
    if atoms_package is not None and hasattr(atoms_package, "signal_processing"):
        delattr(atoms_package, "signal_processing")


def test_merge_creates_package_level_entries_from_atoms_module(tmp_path: Path) -> None:
    workspace = tmp_path
    manifest_path = workspace / "sciona-atoms" / "data" / "audit_manifest.json"
    review_bundle_path = workspace / "sciona-atoms-physics" / "docs" / "review-bundles" / "provider.json"
    provider_module = (
        workspace
        / "sciona-atoms-physics"
        / "src"
        / "sciona"
        / "atoms"
        / "physics"
        / "demo_family"
        / "atoms.py"
    )

    _write(
        provider_module,
        """
        def package_level_atom(values: list[float]) -> list[float]:
            \"\"\"Return values for deterministic testing.\"\"\"
            return values
        """,
    )
    _write(provider_module.parent / "__init__.py", "")
    _write(workspace / "sciona-atoms-physics" / "src" / "sciona" / "__init__.py", "")
    _write(workspace / "sciona-atoms-physics" / "src" / "sciona" / "atoms" / "__init__.py", "")
    _write(workspace / "sciona-atoms-physics" / "src" / "sciona" / "atoms" / "physics" / "__init__.py", "")
    _write(
        manifest_path,
        """
        {
          "schema_version": "1.1",
          "metadata": {},
          "atoms": []
        }
        """,
    )
    _write(
        review_bundle_path,
        """
        {
          "schema_version": "1.0",
          "provider_repo": "sciona-atoms-physics",
          "rows": [
            {
              "atom_name": "sciona.atoms.physics.demo_family.package_level_atom",
              "trust_readiness": "ready_for_manifest_merge",
              "semantic_verdict": "supported",
              "developer_semantic_verdict": "aligned_to_registered_atoms",
              "required_actions": [],
              "limitations": []
            }
          ]
        }
        """,
    )

    summary = merge_audit_manifest_with_review_bundles(
        manifest_path=manifest_path,
        base_dir=workspace,
    )
    merged = json.loads(manifest_path.read_text(encoding="utf-8"))
    atom = merged["atoms"][0]

    assert summary["skipped_unresolved_atom_count"] == 0
    assert atom["atom_name"] == "sciona.atoms.physics.demo_family.package_level_atom"
    assert atom["module_import_path"] == "sciona.atoms.physics.demo_family.atoms"
    assert atom["module_path"].endswith("demo_family/atoms.py")
    assert atom["argument_names"] == ["values"]

    # The fixture creates a regular package solely inside tmp_path. Do not let
    # it mask the real PEP 420 physics provider for later federated tests.
    for module_name in (
        "sciona.atoms.physics.demo_family.atoms",
        "sciona.atoms.physics.demo_family",
        "sciona.atoms.physics",
    ):
        sys.modules.pop(module_name, None)
    atoms_package = sys.modules.get("sciona.atoms")
    if atoms_package is not None and hasattr(atoms_package, "physics"):
        delattr(atoms_package, "physics")
