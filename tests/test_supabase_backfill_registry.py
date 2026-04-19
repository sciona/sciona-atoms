"""Tests for multi-repo registry discovery in supabase backfill."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

from sciona.atoms.supabase_backfill import _registry_paths, load_registry


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


def test_registry_paths_none_discovers_multiple_repos(tmp_path: Path) -> None:
    """When path is None, _registry_paths finds registries in all sibling repos."""
    _write(
        tmp_path / "sciona-atoms" / "data" / "references" / "registry.json",
        '{"schema_version": "1.0", "references": {"ref_a": {"ref_id": "ref_a"}}}',
    )
    _write(
        tmp_path / "sciona-atoms-physics" / "data" / "references" / "registry.json",
        '{"schema_version": "1.0", "references": {"ref_b": {"ref_id": "ref_b"}}}',
    )

    with patch("sciona.atoms.supabase_backfill.provider_repo_roots", return_value=(
        tmp_path / "sciona-atoms",
        tmp_path / "sciona-atoms-physics",
    )):
        paths = _registry_paths(None)

    assert len(paths) == 2
    names = {p.parent.parent.parent.name for p in paths}
    assert names == {"sciona-atoms", "sciona-atoms-physics"}


def test_registry_paths_explicit_returns_single(tmp_path: Path) -> None:
    """When an explicit path is provided, only that one is returned."""
    registry = tmp_path / "custom" / "registry.json"
    _write(registry, '{"references": {}}')

    paths = _registry_paths(registry)
    assert len(paths) == 1
    assert paths[0] == registry.resolve()


def test_load_registry_none_merges_multiple_repos(tmp_path: Path) -> None:
    """load_registry(None) merges entries from all discovered registries."""
    _write(
        tmp_path / "sciona-atoms" / "data" / "references" / "registry.json",
        json.dumps({
            "schema_version": "1.0",
            "references": {
                "shared_ref": {"ref_id": "shared_ref", "title": "Shared Paper"}
            },
        }),
    )
    _write(
        tmp_path / "sciona-atoms-physics" / "data" / "references" / "registry.json",
        json.dumps({
            "schema_version": "1.0",
            "references": {
                "physics_ref": {"ref_id": "physics_ref", "title": "Physics Paper"}
            },
        }),
    )

    with patch("sciona.atoms.supabase_backfill.provider_repo_roots", return_value=(
        tmp_path / "sciona-atoms",
        tmp_path / "sciona-atoms-physics",
    )):
        registry = load_registry(None)

    assert "shared_ref" in registry
    assert "physics_ref" in registry
    assert registry["shared_ref"]["title"] == "Shared Paper"
    assert registry["physics_ref"]["title"] == "Physics Paper"


def test_load_registry_none_detects_conflicts(tmp_path: Path) -> None:
    """load_registry(None) raises on conflicting entries across repos."""
    _write(
        tmp_path / "sciona-atoms" / "data" / "references" / "registry.json",
        json.dumps({
            "references": {"dup": {"ref_id": "dup", "title": "Version A"}}
        }),
    )
    _write(
        tmp_path / "sciona-atoms-bio" / "data" / "references" / "registry.json",
        json.dumps({
            "references": {"dup": {"ref_id": "dup", "title": "Version B"}}
        }),
    )

    with patch("sciona.atoms.supabase_backfill.provider_repo_roots", return_value=(
        tmp_path / "sciona-atoms",
        tmp_path / "sciona-atoms-bio",
    )):
        try:
            load_registry(None)
            assert False, "Expected ValueError for conflicting entries"
        except ValueError as e:
            assert "Conflicting" in str(e)
            assert "dup" in str(e)
