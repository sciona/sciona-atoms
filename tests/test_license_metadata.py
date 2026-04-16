from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "sciona" / "atoms" / "license_metadata.py"


def load_license_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("core_license_metadata", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_discover_repo_license_metadata_prefers_provider_manifest(tmp_path: Path) -> None:
    module = load_license_module()
    repo = tmp_path / "sciona-atoms-signal"
    _write_json(
        repo / "data" / "licenses" / "provider_license.json",
        {
            "schema_version": "1.0",
            "provider_repo": "sciona-atoms-signal",
            "repo_default": {
                "scope": "repo",
                "scope_key": "sciona-atoms-signal",
                "license_expression": "NOASSERTION",
                "license_status": "unknown",
                "license_family": "unknown",
                "source_kind": "manual_override",
                "source_path": None,
                "upstream_license_expression": None,
                "notes": "pending",
            },
            "family_overrides": [
                {
                    "scope": "family",
                    "scope_key": "sciona.atoms.signal_processing.biosppy",
                    "license_expression": "BSD-3-Clause",
                    "license_status": "approved",
                    "license_family": "permissive",
                    "source_kind": "upstream_vendor_license",
                    "source_path": "data/licenses/provider_license.json",
                    "upstream_license_expression": "BSD-3-Clause",
                    "notes": "verified upstream family",
                }
            ],
        },
    )

    metadata = module.discover_repo_license_metadata(repo)
    resolution = module._resolve_for_fqdn(metadata, "sciona.atoms.signal_processing.biosppy.ecg.segment")  # type: ignore[attr-defined]

    assert metadata.repo_name == "sciona-atoms-signal"
    assert resolution.license_expression == "BSD-3-Clause"
    assert resolution.license_status == "approved"
    assert resolution.license_family == "permissive"


def test_discover_repo_license_metadata_falls_back_to_pyproject_license_string(tmp_path: Path) -> None:
    module = load_license_module()
    repo = tmp_path / "sciona-atoms"
    _write(
        repo / "pyproject.toml",
        """
        [project]
        license = "MIT License"
        """,
    )

    metadata = module.discover_repo_license_metadata(repo)

    assert metadata.repo_default.license_expression == "MIT"
    assert metadata.repo_default.license_status == "unknown"
    assert metadata.repo_default.license_source_kind == "pyproject"
    assert metadata.repo_default.license_source_path == "pyproject.toml#project.license"


def test_build_version_license_rows_emits_manifest_backed_rows(tmp_path: Path) -> None:
    module = load_license_module()
    workspace = tmp_path
    repo = workspace / "sciona-atoms"
    _write_json(
        repo / "data" / "licenses" / "provider_license.json",
        {
            "schema_version": "1.0",
            "provider_repo": "sciona-atoms",
            "repo_default": {
                "scope": "repo",
                "scope_key": "sciona-atoms",
                "license_expression": "NOASSERTION",
                "license_status": "unknown",
                "license_family": "unknown",
                "source_kind": "manual_override",
                "source_path": "data/licenses/provider_license.json",
                "upstream_license_expression": None,
                "notes": "pending",
            },
            "family_overrides": [
                {
                    "scope": "family",
                    "scope_key": "sciona.atoms.demo",
                    "license_expression": "Apache License 2.0",
                    "license_status": "approved",
                    "license_family": "permissive",
                    "source_kind": "manual_override",
                    "source_path": "data/licenses/provider_license.json",
                    "upstream_license_expression": "Apache-2.0",
                    "notes": "demo override",
                }
            ],
        },
    )
    _write(
        repo / "src" / "sciona" / "atoms" / "demo" / "ops.py",
        """
        from sciona.ghost.registry import register_atom

        def witness_scale(x):
            return x

        @register_atom(witness_scale)
        def scale(x):
            return x
        """,
    )

    inventory = module.derive_seed_inventory(base_dir=workspace)
    atom_ids = {row.fqdn: f"atom-{index}" for index, row in enumerate(inventory.atom_rows, start=1)}
    version_ids = {
        (row.fqdn, row.content_hash): f"version-{index}"
        for index, row in enumerate(inventory.version_rows, start=1)
    }

    rows, summary = module.build_version_license_rows(
        inventory,
        atom_ids=atom_ids,
        version_ids=version_ids,
        base_dir=workspace,
    )

    assert summary["license_version_rows"] == 1
    assert summary["license_approved_rows"] == 1
    assert summary["license_unknown_rows"] == 0
    assert rows == [
        {
            "atom_id": "atom-1",
            "version_id": "version-1",
            "license_expression": "Apache-2.0",
            "license_status": "approved",
            "license_family": "permissive",
            "license_source_kind": "manual_override",
            "license_source_path": "data/licenses/provider_license.json",
            "upstream_license_expression": "Apache-2.0",
            "license_confidence": "unknown",
            "license_notes": "demo override",
        }
    ]
