from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data/licenses/provider_license.json"


def _load_manifest() -> dict:
    data = json.loads(MANIFEST_PATH.read_text())
    assert data["provider_repo"] == "sciona-atoms"
    assert data["schema_version"] == "1.0"
    return data


def test_core_provider_license_manifest_defaults_to_unknown() -> None:
    manifest = _load_manifest()

    default = manifest["repo_default"]
    assert default["scope"] == "repo"
    assert default["scope_key"] == "sciona-atoms"
    assert default["license_expression"] == "NOASSERTION"
    assert default["license_status"] == "unknown"
    assert default["license_family"] == "unknown"
    assert default["source_kind"] == "manual_override"
    assert default["source_path"] is None
    assert default["upstream_license_expression"] is None
    assert manifest["family_overrides"] == []

