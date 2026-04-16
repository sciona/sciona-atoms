from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from sciona.atoms.audit_review_bundles import (
    discover_review_bundle_paths,
    merge_audit_manifest_with_review_bundles,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


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
    assert signal["review_record_path"] == "signals/review.md"
    assert signal["review_status"] == "missing"
    assert signal["review_required_actions"] == ["add stronger runtime evidence"]
    sys.path.remove(str(package_root))
