from __future__ import annotations

from pathlib import Path

import pytest

from sciona.atoms.provider_inventory import (
    ProviderRepo,
    _default_workspace_root,
    artifact_roots_for_repo,
    discover_shared_data_path,
    discover_provider_repos,
    iter_provider_artifact_files,
    namespace_prefix_for_artifact_root,
)


def test_artifact_roots_for_repo_detects_namespace_layout(tmp_path: Path) -> None:
    namespace_repo = tmp_path / "sciona-atoms"
    (namespace_repo / "src" / "sciona" / "atoms").mkdir(parents=True)

    assert artifact_roots_for_repo(namespace_repo) == (
        (namespace_repo / "src" / "sciona" / "atoms").resolve(),
    )


def test_discover_provider_repos_skips_non_provider_repo_by_default(tmp_path: Path) -> None:
    active_repo = tmp_path / "sciona-atoms"
    addon_repo = tmp_path / "sciona-atoms-ml"
    legacy_repo = tmp_path / "misc-repo"
    active_repo.mkdir()
    addon_repo.mkdir()
    legacy_repo.mkdir()

    repos = discover_provider_repos(tmp_path)

    assert [repo.repo_name for repo in repos] == ["sciona-atoms", "sciona-atoms-ml"]


def test_namespace_prefix_for_artifact_root_supports_namespace_package_paths(tmp_path: Path) -> None:
    namespace_root = tmp_path / "repo" / "src" / "sciona" / "atoms"
    namespace_root.mkdir(parents=True)

    assert namespace_prefix_for_artifact_root(namespace_root) == ("sciona", "atoms")


def test_iter_provider_artifact_files_dedupes_and_skips_pycache(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    (left / "alpha").mkdir(parents=True)
    (left / "__pycache__").mkdir(parents=True)
    (right / "beta").mkdir(parents=True)
    (left / "alpha" / "matches.json").write_text("{}")
    (left / "__pycache__" / "matches.json").write_text("{}")
    (right / "beta" / "matches.json").write_text("{}")

    files = iter_provider_artifact_files("matches.json", roots=(left, right))
    assert files == [
        (left / "alpha" / "matches.json").resolve(),
        (right / "beta" / "matches.json").resolve(),
    ]


def test_discover_shared_data_path_uses_first_matching_provider(tmp_path: Path) -> None:
    provider_a = ProviderRepo("sciona-atoms", tmp_path / "sciona-atoms")
    provider_b = ProviderRepo("sciona-atoms-signal", tmp_path / "sciona-atoms-signal")
    (provider_b.repo_root / "data" / "references").mkdir(parents=True)
    target = provider_b.repo_root / "data" / "references" / "registry.json"
    target.write_text("{}")

    def _provider_roots(_base_dir: Path | None = None) -> tuple[Path, ...]:
        return (provider_a.repo_root.resolve(), provider_b.repo_root.resolve())

    import sciona.atoms.provider_inventory as inventory

    original = inventory.provider_repo_roots
    inventory.provider_repo_roots = _provider_roots  # type: ignore[assignment]
    try:
        resolved = discover_shared_data_path(Path("data/references/registry.json"))
    finally:
        inventory.provider_repo_roots = original  # type: ignore[assignment]

    assert resolved == target.resolve()


def test_discover_shared_data_path_falls_back_to_sciona_atoms_when_no_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    import sciona.atoms.provider_inventory as inventory

    monkeypatch.setattr(inventory, "provider_repo_roots", lambda _base_dir=None: tuple())

    resolved = discover_shared_data_path(Path("data/references/registry.json"))

    assert resolved == (inventory._DEFAULT_WORKSPACE_ROOT / "sciona-atoms" / "data/references/registry.json").resolve()


def test_default_workspace_root_uses_repo_parent() -> None:
    module_root = Path("/workspace/sciona-atoms/src/sciona/atoms/provider_inventory.py")
    assert _default_workspace_root(module_root) == Path("/workspace").resolve()
