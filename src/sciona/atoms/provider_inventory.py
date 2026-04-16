"""Shared provider/artifact discovery for Supabase population tooling.

This module is intended to become the canonical home for sibling provider-repo
discovery and file-backed artifact enumeration used to populate Supabase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

def _default_workspace_root(module_root: Path) -> Path:
    """Infer the sibling-repo workspace root from this module path."""
    return module_root.parents[4].resolve()


_MODULE_ROOT = Path(__file__).resolve()
_DEFAULT_WORKSPACE_ROOT = _default_workspace_root(_MODULE_ROOT)
_ARTIFACT_ROOT_CANDIDATES: tuple[Path, ...] = (
    Path("src/sciona/atoms"),
    Path("sciona/atoms"),
)
_AUDIT_MANIFEST_RELATIVE = Path("data/audit_manifest.json")
_AUDIT_REVIEW_BUNDLE_DIRS: tuple[Path, ...] = (
    Path("data/audit_reviews"),
    Path("data/review_bundles"),
    Path("docs/review-bundles"),
)
_LICENSE_SOURCE_CANDIDATES: tuple[Path, ...] = (
    Path("pyproject.toml"),
    Path("LICENSE"),
    Path("LICENSE.md"),
    Path("COPYING"),
    Path("COPYING.md"),
    Path("NOTICE"),
    Path("NOTICE.md"),
)
_LICENSE_MANIFEST_CANDIDATES: tuple[Path, ...] = (
    Path("data/licenses/provider_license.json"),
    Path("docs/license-manifest.json"),
    Path("license_manifest.json"),
)
_REFERENCES_REGISTRY_RELATIVE = Path("data/references/registry.json")
_PROVIDER_REPO_ORDER: tuple[str, ...] = (
    "sciona-atoms",
    "sciona-atoms-bio",
    "sciona-atoms-cs",
    "sciona-atoms-fintech",
    "sciona-atoms-ml",
    "sciona-atoms-physics",
    "sciona-atoms-robotics",
    "sciona-atoms-signal",
)
_NAMESPACE_ANCHORS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("src", "sciona", "atoms"), ("sciona", "atoms")),
    (("sciona", "atoms"), ("sciona", "atoms")),
)


@dataclass(frozen=True)
class ProviderRepo:
    repo_name: str
    repo_root: Path


def _dedupe_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return tuple(deduped)


def _find_anchor(parts: Sequence[str], anchor: Sequence[str]) -> int | None:
    width = len(anchor)
    if width == 0 or len(parts) < width:
        return None
    for index in range(len(parts) - width + 1):
        if tuple(parts[index : index + width]) == tuple(anchor):
            return index
    return None


def _repo_order_key(name: str) -> tuple[int, str]:
    try:
        return (_PROVIDER_REPO_ORDER.index(name), name)
    except ValueError:
        return (len(_PROVIDER_REPO_ORDER), name)


def workspace_root() -> Path:
    """Return the workspace root that contains sibling provider repos."""
    configured = str(os.environ.get("SCIONA_PROVIDER_WORKSPACE_ROOT", "") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return _DEFAULT_WORKSPACE_ROOT.resolve()


def discover_provider_repos(base_dir: Path | None = None) -> tuple[ProviderRepo, ...]:
    """Return sibling provider repositories in deterministic order."""
    configured = str(os.environ.get("SCIONA_ATOM_PROVIDER_ROOTS", "") or "").strip()
    if configured:
        roots = _dedupe_paths(
            Path(token)
            for token in configured.split(os.pathsep)
            if str(token).strip()
        )
        return tuple(
            ProviderRepo(repo_name=root.name, repo_root=root)
            for root in roots
        )

    parent = (base_dir or workspace_root()).expanduser().resolve()
    repos: list[ProviderRepo] = []
    for child in sorted(parent.iterdir(), key=lambda path: _repo_order_key(path.name)):
        if not child.is_dir():
            continue
        if child.name == "sciona-atoms" or child.name.startswith("sciona-atoms-"):
            repos.append(ProviderRepo(repo_name=child.name, repo_root=child.resolve()))
    return tuple(repos)


def provider_repo_roots(base_dir: Path | None = None) -> tuple[Path, ...]:
    """Return the resolved repository roots for all discovered providers."""
    return tuple(repo.repo_root for repo in discover_provider_repos(base_dir))


def artifact_roots_for_repo(repo_root: Path) -> tuple[Path, ...]:
    """Return artifact-bearing roots within a provider repo."""
    roots: list[Path] = []
    for relative in _ARTIFACT_ROOT_CANDIDATES:
        candidate = (repo_root / relative).resolve()
        if candidate.exists():
            roots.append(candidate)
    return _dedupe_paths(roots)


def discover_artifact_roots(base_dir: Path | None = None) -> tuple[Path, ...]:
    """Return all artifact roots across discovered providers."""
    roots: list[Path] = []
    for repo_root in provider_repo_roots(base_dir):
        roots.extend(artifact_roots_for_repo(repo_root))
    return _dedupe_paths(roots)


def artifact_root_namespace_prefix(root: Path) -> tuple[str, ...] | None:
    """Return the namespace prefix when ``root`` exactly matches a configured anchor."""
    parts = root.resolve().parts
    for anchor, prefix in _NAMESPACE_ANCHORS:
        index = _find_anchor(parts, anchor)
        if index is not None and index + len(anchor) == len(parts):
            return prefix
    return None


def namespace_prefix_for_artifact_root(root: Path) -> tuple[str, ...]:
    """Return the dotted import namespace implied by an artifact root."""
    prefix = artifact_root_namespace_prefix(root)
    if prefix is not None:
        return prefix
    return (root.name,)


def iter_provider_artifact_files(
    filename: str,
    *,
    roots: Sequence[Path] | None = None,
    base_dir: Path | None = None,
) -> list[Path]:
    """Return matching artifact files across all configured provider roots."""
    search_roots = tuple(roots) if roots is not None else discover_artifact_roots(base_dir)
    matches: list[Path] = []
    for root in search_roots:
        for path in sorted(root.rglob(filename)):
            if "__pycache__" in path.parts:
                continue
            matches.append(path.resolve())
    return sorted(_dedupe_paths(matches))


def discover_shared_data_path(
    relative_path: str | Path,
    *,
    env_var: str | None = None,
    base_dir: Path | None = None,
) -> Path:
    """Return the first shared data file found across discovered providers."""
    if env_var:
        configured = str(os.environ.get(env_var, "") or "").strip()
        if configured:
            return Path(configured).expanduser().resolve()

    relative = Path(relative_path)
    for repo_root in provider_repo_roots(base_dir):
        candidate = (repo_root / relative).resolve()
        if candidate.exists():
            return candidate

    fallback_root = provider_repo_roots(base_dir)
    if fallback_root:
        return (fallback_root[-1] / relative).resolve()
    return (_DEFAULT_WORKSPACE_ROOT / "sciona-atoms" / relative).resolve()


def discover_audit_manifest_path(base_dir: Path | None = None) -> Path:
    """Return the canonical audit manifest path."""
    return discover_shared_data_path(
        _AUDIT_MANIFEST_RELATIVE,
        env_var="AUDIT_MANIFEST_PATH",
        base_dir=base_dir,
    )


def discover_audit_review_bundle_paths(base_dir: Path | None = None) -> tuple[Path, ...]:
    """Return all provider-owned audit review bundle files in deterministic order."""
    bundle_paths: list[Path] = []
    for repo_root in provider_repo_roots(base_dir):
        for relative in _AUDIT_REVIEW_BUNDLE_DIRS:
            review_dir = (repo_root / relative).resolve()
            if not review_dir.is_dir():
                continue
            for path in sorted(review_dir.rglob("*.json")):
                if path.is_file():
                    bundle_paths.append(path.resolve())
        src_root = (repo_root / "src").resolve()
        if src_root.is_dir():
            for path in sorted(src_root.rglob("review_bundle.json")):
                if path.is_file():
                    bundle_paths.append(path.resolve())
    return _dedupe_paths(bundle_paths)


def discover_license_source_paths(base_dir: Path | None = None) -> tuple[Path, ...]:
    """Return provider-owned license source files in deterministic order."""
    source_paths: list[Path] = []
    for repo_root in provider_repo_roots(base_dir):
        for relative in _LICENSE_SOURCE_CANDIDATES:
            candidate = (repo_root / relative).resolve()
            if candidate.is_file():
                source_paths.append(candidate)
    return _dedupe_paths(source_paths)


def discover_license_manifest_paths(base_dir: Path | None = None) -> tuple[Path, ...]:
    """Return provider-owned license manifests in deterministic order."""
    manifest_paths: list[Path] = []
    for repo_root in provider_repo_roots(base_dir):
        for relative in _LICENSE_MANIFEST_CANDIDATES:
            candidate = (repo_root / relative).resolve()
            if candidate.is_file():
                manifest_paths.append(candidate)
    return _dedupe_paths(manifest_paths)


def discover_references_registry_path(base_dir: Path | None = None) -> Path:
    """Return the canonical references registry path."""
    return discover_shared_data_path(
        _REFERENCES_REGISTRY_RELATIVE,
        env_var="REFERENCES_REGISTRY_PATH",
        base_dir=base_dir,
    )
