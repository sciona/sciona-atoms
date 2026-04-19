#!/usr/bin/env python3
"""Migrate the shared references registry into per-repo registries.

Scans each sibling sciona-atoms-* repo to find which ref_ids are used by
its per-atom references.json files, then writes a local
data/references/registry.json in each repo containing only the entries
that repo needs. The shared sciona-atoms registry is trimmed to only
entries used by sciona-atoms itself plus any unused entries (home of
last resort).

Idempotent — running twice produces the same result. Merges with existing
local registries rather than overwriting.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent.parent
SHARED_REGISTRY = WORKSPACE / "sciona-atoms" / "data" / "references" / "registry.json"


def discover_repos() -> list[Path]:
    """Find all sciona-atoms* repos in the workspace."""
    repos = []
    for child in sorted(WORKSPACE.iterdir()):
        if child.is_dir() and (child.name == "sciona-atoms" or child.name.startswith("sciona-atoms-")):
            repos.append(child)
    return repos


def scan_used_ref_ids(repo: Path) -> set[str]:
    """Find all ref_ids referenced by per-atom references.json files in a repo."""
    used: set[str] = set()
    src_dir = repo / "src"
    if not src_dir.exists():
        return used
    for ref_file in sorted(src_dir.rglob("references.json")):
        if "__pycache__" in ref_file.parts:
            continue
        try:
            data = json.loads(ref_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for atom_entry in data.get("atoms", {}).values():
            for ref in atom_entry.get("references", []):
                ref_id = ref.get("ref_id", "").strip()
                if ref_id:
                    used.add(ref_id)
    return used


def load_existing_registry(repo: Path) -> dict[str, dict]:
    """Load an existing local registry, or return empty dict."""
    path = repo / "data" / "references" / "registry.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("references", {})
    except (json.JSONDecodeError, OSError):
        return {}


def write_registry(repo: Path, entries: dict[str, dict]) -> None:
    """Write a registry.json to a repo's data/references/ directory."""
    path = repo / "data" / "references" / "registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "1.0", "references": dict(sorted(entries.items()))}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    if not SHARED_REGISTRY.exists():
        print(f"ERROR: shared registry not found at {SHARED_REGISTRY}", file=sys.stderr)
        return 1

    shared = json.loads(SHARED_REGISTRY.read_text(encoding="utf-8"))
    all_refs: dict[str, dict] = shared.get("references", {})
    print(f"Shared registry: {len(all_refs)} entries")

    repos = discover_repos()
    print(f"Found {len(repos)} repos: {[r.name for r in repos]}")

    # Scan usage
    repo_usage: dict[str, set[str]] = {}
    for repo in repos:
        repo_usage[repo.name] = scan_used_ref_ids(repo)

    used_anywhere: set[str] = set()
    for ids in repo_usage.values():
        used_anywhere |= ids

    # Write per-repo registries
    for repo in repos:
        needed_ids = repo_usage[repo.name]
        if repo.name == "sciona-atoms":
            # Keep entries used by sciona-atoms + unused entries (home of last resort)
            unused = set(all_refs.keys()) - used_anywhere
            needed_ids = needed_ids | unused

        # Build the entries for this repo
        entries: dict[str, dict] = {}
        for ref_id in sorted(needed_ids):
            if ref_id in all_refs:
                entries[ref_id] = all_refs[ref_id]

        # For sibling repos, merge with existing local registry to preserve
        # local-only entries (e.g. physics already has TrackML refs).
        # For sciona-atoms, DON'T merge — we're trimming the shared registry.
        if repo.name != "sciona-atoms":
            existing = load_existing_registry(repo)
            for ref_id, entry in existing.items():
                if ref_id not in entries:
                    entries[ref_id] = entry

        if not entries:
            print(f"  {repo.name}: 0 entries (skipping)")
            continue

        write_registry(repo, entries)
        print(f"  {repo.name}: {len(entries)} entries written")

    # Verify: load_registry(None) should work without conflicts
    print("\nVerifying merged registry...")
    total_ids: dict[str, list[str]] = defaultdict(list)
    for repo in repos:
        local = load_existing_registry(repo)
        for ref_id, entry in local.items():
            total_ids[ref_id].append(repo.name)

    conflicts = 0
    for ref_id, sources in total_ids.items():
        if len(sources) > 1:
            # Check if entries are identical
            entries_by_repo = {}
            for repo in repos:
                if repo.name in sources:
                    local = load_existing_registry(repo)
                    if ref_id in local:
                        entries_by_repo[repo.name] = local[ref_id]
            values = list(entries_by_repo.values())
            if not all(v == values[0] for v in values):
                print(f"  CONFLICT: {ref_id} differs across {sources}")
                conflicts += 1

    total_unique = len(total_ids)
    print(f"\nTotal unique ref_ids across all repos: {total_unique}")
    print(f"Conflicts: {conflicts}")
    return 1 if conflicts > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
