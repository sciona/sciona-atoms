"""Provider-owned discovery and seeding for version-scoped license metadata."""

from __future__ import annotations

import argparse
import json
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from sciona.atoms.provider_inventory import discover_provider_repos
from sciona.atoms.supabase_seed import SeedInventory, derive_seed_inventory

_SPDX_OPERATORS = {"AND", "OR", "WITH"}
_SPDX_TOKEN_RE = re.compile(r"\(|\)|AND|OR|WITH|[A-Za-z0-9][A-Za-z0-9.+-]*")
_SPDX_ALIASES = {
    "apache 2.0": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "bsd 2-clause": "BSD-2-Clause",
    "bsd 3-clause": "BSD-3-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "isc": "ISC",
    "mit": "MIT",
    "mit license": "MIT",
    "mpl-2.0": "MPL-2.0",
    "mozilla public license 2.0": "MPL-2.0",
    "cc-by-4.0": "CC-BY-4.0",
    "creative commons attribution 4.0 international": "CC-BY-4.0",
    "gpl-3.0-only": "GPL-3.0-only",
    "gpl v3": "GPL-3.0-only",
    "lgpl-2.1-only": "LGPL-2.1-only",
    "lgpl v2.1": "LGPL-2.1-only",
    "noassertion": "NOASSERTION",
}
_LICENSE_FILE_CANDIDATES = (
    "LICENSE",
    "LICENSE.md",
    "COPYING",
    "COPYING.md",
    "NOTICE",
    "NOTICE.md",
)
_MANIFEST_CANDIDATES = (
    Path("data/licenses/provider_license.json"),
    Path("docs/license-manifest.json"),
    Path("license_manifest.json"),
)


@dataclass(frozen=True)
class LicenseResolution:
    license_expression: str
    license_status: str
    license_family: str
    license_source_kind: str
    license_source_path: str
    upstream_license_expression: str
    license_confidence: str
    license_notes: str = ""


@dataclass(frozen=True)
class RepoLicenseMetadata:
    repo_name: str
    repo_root: Path
    repo_default: LicenseResolution
    family_overrides: tuple[tuple[str, LicenseResolution], ...] = ()


@dataclass(frozen=True)
class VersionLicenseSeedRow:
    fqdn: str
    version_id: str
    license_expression: str
    license_status: str
    license_family: str
    license_source_kind: str
    license_source_path: str
    upstream_license_expression: str
    license_confidence: str
    license_notes: str = ""

    def as_dict(self, *, atom_id: str) -> dict[str, Any]:
        return {
            "atom_id": atom_id,
            "version_id": self.version_id,
            "license_expression": self.license_expression,
            "license_status": self.license_status,
            "license_family": self.license_family,
            "license_source_kind": self.license_source_kind,
            "license_source_path": self.license_source_path,
            "upstream_license_expression": self.upstream_license_expression,
            "license_confidence": self.license_confidence,
            "license_notes": self.license_notes,
        }


def _dedupe_preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _coalesce_text(mapping: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_spdx_like_expression(raw: str) -> str:
    """Return a deterministic SPDX-like normalization for a raw license string."""
    text = " ".join(str(raw or "").split()).strip()
    if not text:
        return ""
    alias = _SPDX_ALIASES.get(text.lower())
    if alias:
        return alias
    tokens = _SPDX_TOKEN_RE.findall(text)
    if not tokens:
        return ""
    normalized: list[str] = []
    for token in tokens:
        if token in {"(", ")"}:
            normalized.append(token)
            continue
        upper = token.upper()
        if upper in _SPDX_OPERATORS:
            normalized.append(upper)
            continue
        alias = _SPDX_ALIASES.get(token.lower())
        if alias:
            normalized.append(alias)
            continue
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+-]*", token):
            normalized.append(token)
            continue
        return ""
    rendered = " ".join(normalized)
    rendered = rendered.replace("( ", "(").replace(" )", ")")
    rendered = re.sub(r"\s+", " ", rendered).strip()
    return rendered


def _license_family(expression: str) -> str:
    normalized = normalize_spdx_like_expression(expression)
    if not normalized or normalized == "NOASSERTION":
        return "unknown"
    if any(token in normalized for token in ("GPL-", "AGPL-")):
        return "strong_copyleft"
    if any(token in normalized for token in ("LGPL-", "MPL-")):
        return "weak_copyleft"
    if any(token in normalized for token in ("Commercial", "Proprietary")):
        return "proprietary"
    return "permissive"


def _normalize_status(raw_status: str) -> str:
    lowered = str(raw_status or "").strip().lower()
    if lowered in {"approved", "verified", "resolved"}:
        return "approved"
    if lowered in {"restricted", "blocked"}:
        return "restricted"
    if lowered in {"needs_legal_review", "review_required"}:
        return "needs_legal_review"
    return "unknown"


def _normalize_resolution(
    *,
    expression: str,
    status: str,
    family: str = "",
    source_kind: str = "",
    source_path: str = "",
    upstream_expression: str = "",
    confidence: str = "",
    notes: str = "",
) -> LicenseResolution:
    normalized_expression = normalize_spdx_like_expression(expression) or expression.strip()
    normalized_upstream = normalize_spdx_like_expression(upstream_expression) or upstream_expression.strip()
    normalized_status = _normalize_status(status)
    normalized_family = family.strip() or _license_family(normalized_expression)
    normalized_source_kind = source_kind.strip() or "unknown"
    normalized_confidence = confidence.strip() or "unknown"
    return LicenseResolution(
        license_expression=normalized_expression,
        license_status=normalized_status,
        license_family=normalized_family or "unknown",
        license_source_kind=normalized_source_kind,
        license_source_path=source_path.strip(),
        upstream_license_expression=normalized_upstream,
        license_confidence=normalized_confidence,
        license_notes=notes.strip(),
    )


def _detect_license_from_text(text: str) -> tuple[str, str]:
    lowered = " ".join(text.lower().split())
    if "mit license" in lowered or "permission is hereby granted" in lowered:
        return "MIT", "high"
    if "apache license" in lowered and "version 2.0" in lowered:
        return "Apache-2.0", "high"
    if "redistribution and use in source and binary forms" in lowered and "neither the name" in lowered:
        return "BSD-3-Clause", "high"
    if "redistribution and use in source and binary forms" in lowered:
        return "BSD-2-Clause", "medium"
    if "isc license" in lowered:
        return "ISC", "high"
    if "creative commons attribution 4.0 international" in lowered or "cc-by-4.0" in lowered:
        return "CC-BY-4.0", "medium"
    return "", "unknown"


def _load_pyproject_license(pyproject_path: Path) -> LicenseResolution | None:
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = payload.get("project", {}) if isinstance(payload, dict) else {}
    license_spec = project.get("license") if isinstance(project, dict) else None
    if isinstance(license_spec, str):
        return _normalize_resolution(
            expression=license_spec,
            status="unknown",
            source_kind="pyproject",
            source_path="pyproject.toml#project.license",
            confidence="high",
        )
    if isinstance(license_spec, dict):
        license_text = license_spec.get("text")
        if isinstance(license_text, str) and license_text.strip():
            return _normalize_resolution(
                expression=license_text,
                status="unknown",
                source_kind="pyproject",
                source_path="pyproject.toml#project.license",
                confidence="high",
            )
        license_file = license_spec.get("file")
        if isinstance(license_file, str) and license_file.strip():
            candidate = (pyproject_path.parent / license_file).resolve()
            if candidate.is_file():
                expr, confidence = _detect_license_from_text(candidate.read_text(encoding="utf-8"))
                if expr:
                    return _normalize_resolution(
                        expression=expr,
                        status="unknown",
                        source_kind="license_file",
                        source_path=license_file,
                        confidence=confidence,
                    )
    return None


def _discover_repo_license_file(repo_root: Path) -> Path | None:
    for name in _LICENSE_FILE_CANDIDATES:
        candidate = (repo_root / name).resolve()
        if candidate.is_file():
            return candidate
    return None


def _fallback_repo_resolution(repo_root: Path) -> LicenseResolution:
    pyproject = (repo_root / "pyproject.toml").resolve()
    if pyproject.is_file():
        resolved = _load_pyproject_license(pyproject)
        if resolved is not None:
            return resolved
    license_file = _discover_repo_license_file(repo_root)
    if license_file is not None:
        expr, confidence = _detect_license_from_text(license_file.read_text(encoding="utf-8"))
        if expr:
            return _normalize_resolution(
                expression=expr,
                status="unknown",
                source_kind="license_file",
                source_path=license_file.relative_to(repo_root).as_posix(),
                confidence=confidence,
            )
    return _normalize_resolution(
        expression="NOASSERTION",
        status="unknown",
        source_kind="unknown",
        source_path="",
        confidence="unknown",
        notes="No provider license manifest or authoritative repo-level license declaration found.",
    )


def _discover_manifest_path(repo_root: Path) -> Path | None:
    for relative in _MANIFEST_CANDIDATES:
        candidate = (repo_root / relative).resolve()
        if candidate.is_file():
            return candidate
    return None


def _parse_standard_entry(entry: dict[str, Any], *, fallback_path: str) -> tuple[str, LicenseResolution]:
    scope_key = _coalesce_text(entry, "scope_key", "family") or "repo"
    expression = _coalesce_text(entry, "license_expression", "license_spdx") or "NOASSERTION"
    status = _coalesce_text(entry, "license_status", "status") or "unknown"
    family = _coalesce_text(entry, "license_family")
    source_kind = _coalesce_text(entry, "source_kind") or "manual_override"
    source_path = _coalesce_text(entry, "source_path", "review_record_path") or fallback_path
    upstream = _coalesce_text(entry, "upstream_license_expression")
    confidence = _coalesce_text(entry, "confidence") or "unknown"
    notes = _coalesce_text(entry, "notes", "reason", "rationale")
    return scope_key, _normalize_resolution(
        expression=expression,
        status=status,
        family=family,
        source_kind=source_kind,
        source_path=source_path,
        upstream_expression=upstream,
        confidence=confidence,
        notes=notes,
    )


def _load_provider_manifest(repo_root: Path) -> RepoLicenseMetadata | None:
    manifest_path = _discover_manifest_path(repo_root)
    if manifest_path is None:
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_name = str(payload.get("provider_repo") or repo_root.name)
    relative_manifest_path = manifest_path.relative_to(repo_root).as_posix()

    repo_default_payload = (
        payload.get("repo_default")
        or payload.get("repository_default")
        or payload.get("repo_default_license")
        or {}
    )
    _scope_key, repo_default = _parse_standard_entry(
        repo_default_payload,
        fallback_path=relative_manifest_path,
    )

    overrides: list[tuple[str, LicenseResolution]] = []
    for entry in payload.get("family_overrides", []) or []:
        if not isinstance(entry, dict):
            continue
        overrides.append(_parse_standard_entry(entry, fallback_path=relative_manifest_path))

    for entry in payload.get("family_inventory", []) or []:
        if not isinstance(entry, dict):
            continue
        overrides.append(_parse_standard_entry(entry, fallback_path=relative_manifest_path))

    overrides = sorted(
        [(scope_key, resolution) for scope_key, resolution in overrides if scope_key],
        key=lambda item: (-len(item[0]), item[0]),
    )
    return RepoLicenseMetadata(
        repo_name=repo_name,
        repo_root=repo_root,
        repo_default=repo_default,
        family_overrides=tuple(overrides),
    )


def discover_repo_license_metadata(repo_root: Path) -> RepoLicenseMetadata:
    repo_root = repo_root.expanduser().resolve()
    manifest = _load_provider_manifest(repo_root)
    if manifest is not None:
        return manifest
    return RepoLicenseMetadata(
        repo_name=repo_root.name,
        repo_root=repo_root,
        repo_default=_fallback_repo_resolution(repo_root),
        family_overrides=tuple(),
    )


def discover_provider_license_metadata(base_dir: Path | None = None) -> tuple[RepoLicenseMetadata, ...]:
    metadata: list[RepoLicenseMetadata] = []
    for repo in discover_provider_repos(base_dir):
        metadata.append(discover_repo_license_metadata(repo.repo_root))
    return tuple(sorted(metadata, key=lambda row: row.repo_name))


def _resolve_for_fqdn(metadata: RepoLicenseMetadata, fqdn: str) -> LicenseResolution:
    for scope_key, resolution in metadata.family_overrides:
        if fqdn.startswith(scope_key):
            return resolution
    return metadata.repo_default


def build_version_license_rows(
    inventory: SeedInventory,
    *,
    atom_ids: dict[str, str],
    version_ids: dict[tuple[str, str], str],
    base_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    repo_license_by_name = {
        row.repo_name: row for row in discover_provider_license_metadata(base_dir)
    }
    atom_repo_by_fqdn = {row.fqdn: row.repo_name for row in inventory.atom_rows}
    rows: list[dict[str, Any]] = []
    approved_rows = 0
    unknown_rows = 0
    restricted_rows = 0
    legal_review_rows = 0
    missing_atom = 0
    missing_version = 0

    for version in inventory.version_rows:
        atom_id = atom_ids.get(version.fqdn)
        if atom_id is None:
            missing_atom += 1
            continue
        version_id = version_ids.get((version.fqdn, version.content_hash))
        if version_id is None:
            missing_version += 1
            continue
        repo_name = atom_repo_by_fqdn.get(version.fqdn, "")
        repo_license = repo_license_by_name.get(repo_name)
        resolution = _resolve_for_fqdn(repo_license, version.fqdn) if repo_license else _normalize_resolution(
            expression="NOASSERTION",
            status="unknown",
            source_kind="unknown",
            confidence="unknown",
            notes="No provider license metadata found.",
        )
        row = VersionLicenseSeedRow(
            fqdn=version.fqdn,
            version_id=version_id,
            license_expression=resolution.license_expression,
            license_status=resolution.license_status,
            license_family=resolution.license_family,
            license_source_kind=resolution.license_source_kind,
            license_source_path=resolution.license_source_path,
            upstream_license_expression=resolution.upstream_license_expression,
            license_confidence=resolution.license_confidence,
            license_notes=resolution.license_notes,
        )
        rows.append(row.as_dict(atom_id=atom_id))
        if resolution.license_status == "approved":
            approved_rows += 1
        elif resolution.license_status == "restricted":
            restricted_rows += 1
        elif resolution.license_status == "needs_legal_review":
            legal_review_rows += 1
        else:
            unknown_rows += 1

    rows.sort(key=lambda row: (row["version_id"], row["atom_id"]))
    summary = {
        "license_version_rows": len(rows),
        "license_approved_rows": approved_rows,
        "license_unknown_rows": unknown_rows,
        "license_restricted_rows": restricted_rows,
        "license_needs_legal_review_rows": legal_review_rows,
        "license_missing_atom": missing_atom,
        "license_missing_version": missing_version,
    }
    return rows, summary


def _upsert_rows(client: Any, table: str, rows: Sequence[dict[str, Any]], *, conflict: str) -> None:
    if not rows:
        return
    client.table(table).upsert(list(rows), on_conflict=conflict).execute()


def _select_all_rows(client: Any, table: str, columns: str, *, page_size: int = 1000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = 0
    while True:
        response = client.table(table).select(columns).range(start, start + page_size - 1).execute()
        batch = list(getattr(response, "data", None) or [])
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return rows


def _fetch_atom_ids(client: Any) -> dict[str, str]:
    rows = _select_all_rows(client, "atoms", "atom_id, fqdn")
    return {row["fqdn"]: row["atom_id"] for row in rows if row.get("fqdn") and row.get("atom_id")}


def seed_atom_version_license_metadata(
    client: Any,
    inventory: SeedInventory | None = None,
    *,
    base_dir: Path | None = None,
    atom_ids: dict[str, str] | None = None,
    version_ids: dict[tuple[str, str], str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    inventory = inventory or derive_seed_inventory(base_dir=base_dir)
    if atom_ids is None:
        atom_ids = {row.fqdn: row.fqdn for row in inventory.atom_rows} if dry_run else _fetch_atom_ids(client)
    version_ids = version_ids or {(row.fqdn, row.content_hash): row.version_id for row in inventory.version_rows}
    rows, summary = build_version_license_rows(
        inventory,
        atom_ids=atom_ids,
        version_ids=version_ids,
        base_dir=base_dir,
    )
    summary = dict(summary)
    summary["dry_run"] = dry_run
    if dry_run:
        return summary
    _upsert_rows(client, "atom_version_license_metadata", rows, conflict="version_id")
    summary["applied"] = True
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply provider-owned license metadata rows.")
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    inventory = derive_seed_inventory(base_dir=args.workspace_root)
    summary = {
        "provider_repos": len(inventory.provider_repos),
        "version_rows": len(inventory.version_rows),
    }
    if args.apply:
        client = create_supabase_client_from_env()
        apply_summary = seed_atom_version_license_metadata(
            client,
            inventory,
            base_dir=args.workspace_root,
        )
        summary.update(apply_summary)
    else:
        atom_ids = {row.fqdn: row.fqdn for row in inventory.atom_rows}
        version_ids = {(row.fqdn, row.content_hash): row.version_id for row in inventory.version_rows}
        rows, row_summary = build_version_license_rows(
            inventory,
            atom_ids=atom_ids,
            version_ids=version_ids,
            base_dir=args.workspace_root,
        )
        summary.update(row_summary)
        summary["sample_rows"] = rows[:3]
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


def create_supabase_client_from_env() -> Any:
    from supabase import create_client

    service_key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_SERVICE_KEY")
        or os.environ.get("SCIONA_SUPABASE_SERVICE_ROLE_KEY")
        or os.environ["SCIONA_SUPABASE_SERVICE_KEY"]
    )
    url = os.environ.get("SUPABASE_URL") or os.environ["SCIONA_SUPABASE_URL"]
    return create_client(url, service_key)
