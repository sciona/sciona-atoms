#!/usr/bin/env python3
"""Print provider/artifact inventory used by Supabase population tooling."""

from __future__ import annotations

import json
from pathlib import Path

from sciona.atoms.provider_inventory import (
    artifact_roots_for_repo,
    discover_audit_manifest_path,
    discover_provider_repos,
    discover_references_registry_path,
)


def main() -> int:
    payload = {
        "workspace_root": str(Path(__file__).resolve().parents[2]),
        "providers": [
            {
                "repo_name": provider.repo_name,
                "repo_root": str(provider.repo_root),
                "artifact_roots": [str(path) for path in artifact_roots_for_repo(provider.repo_root)],
            }
            for provider in discover_provider_repos()
        ],
        "audit_manifest_path": str(discover_audit_manifest_path()),
        "references_registry_path": str(discover_references_registry_path()),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
