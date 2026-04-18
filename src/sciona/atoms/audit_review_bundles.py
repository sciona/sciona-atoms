"""Provider-owned merge workflow for audit review bundle files.

This merger accepts both a compact ``atoms`` schema and richer provider-owned
family batch bundles that expose ``rows`` with one or more atom identifiers.
When a bundle introduces an atom that is not yet present in the central audit
manifest, the merger derives a base manifest entry directly from the installed
callable so the existing parameter/description/IO backfills can still operate.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sciona.atoms.provider_inventory import (
    discover_audit_manifest_path,
    discover_audit_review_bundle_paths,
)

AUDIT_BUNDLE_SCHEMA_VERSION = "1.0"
_IDENTIFIER_FIELDS = {"atom_name", "atom_id", "atom_key"}
_TOP_LEVEL_METADATA_FIELDS = {"schema_version", "provider_repo", "bundle_name", "generated_at"}
_ROW_CONTROL_FIELDS = {
    "authoritative_sources",
    "blocking_findings",
    "developer_semantic_verdict",
    "limitations",
    "required_actions",
    "review_developer_semantic_verdict",
    "review_developer_semantics_verdict",
    "review_record_path",
    "review_semantic_verdict",
    "review_status",
    "semantic_verdict",
    "source_paths",
    "trust_readiness",
}
_READY_TRUST_STATES = {"ready", "catalog_ready", "ready_for_manifest_merge"}
_PASS_VERDICTS = {"pass", "supported", "aligned_to_registered_atoms"}
_REVIEW_DIR_HINTS = (
    Path("data/audit_reviews"),
    Path("data/review_bundles"),
    Path("docs/review-bundles"),
)


@dataclass(frozen=True)
class ReviewBundleEntry:
    atom_name: str
    patch: dict[str, Any]
    source_path: Path
    record_path: str
    atom_key: str | None = None


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in {path}")
    return [item for item in payload if isinstance(item, dict)]


def _flatten_bundle_entry(raw_entry: dict[str, Any], *, source_path: Path, record_path: str) -> ReviewBundleEntry:
    atom_name = str(raw_entry.get("atom_name") or "").strip()
    if not atom_name:
        raise ValueError(f"Review bundle entry in {source_path} is missing atom_name")
    patch: dict[str, Any] = {}
    for key, value in raw_entry.items():
        if key in _IDENTIFIER_FIELDS or key in _TOP_LEVEL_METADATA_FIELDS:
            continue
        if key == "audit" and isinstance(value, dict):
            for audit_key, audit_value in value.items():
                if audit_key not in _IDENTIFIER_FIELDS and audit_value is not None:
                    patch[audit_key] = audit_value
            continue
        if value is not None:
            patch[key] = value
    if not patch:
        raise ValueError(f"Review bundle entry for {atom_name} in {source_path} has no mergeable fields")
    return ReviewBundleEntry(atom_name=atom_name, patch=patch, source_path=source_path, record_path=record_path)


def discover_review_bundle_paths(base_dir: Path | None = None) -> tuple[Path, ...]:
    """Return all provider-owned review bundle files in deterministic order."""
    return discover_audit_review_bundle_paths(base_dir)


def load_review_bundle_entries(path: Path) -> list[ReviewBundleEntry]:
    """Load and validate a single review bundle file."""
    payload = _load_json_object(path)
    schema_version = str(payload.get("schema_version") or payload.get("bundle_version") or "").strip()
    if schema_version != AUDIT_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported review bundle schema version {schema_version!r} in {path}")

    resolved = path.resolve()
    parts = resolved.parts
    try:
        data_index = parts.index("data")
        repo_root = Path(*parts[:data_index])
    except ValueError:
        repo_root = resolved.parent
    try:
        record_path = str(path.relative_to(repo_root))
    except Exception:
        record_path = str(path)

    atoms = payload.get("atoms")
    if isinstance(atoms, list):
        entries = [
            _flatten_bundle_entry(atom, source_path=path, record_path=record_path)
            for atom in atoms
            if isinstance(atom, dict)
        ]
        return sorted(entries, key=lambda entry: entry.atom_name)

    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Expected atoms or rows list in {path}")
    entries = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        entries.extend(
            _flatten_row_bundle_entries(
                payload,
                row,
                source_path=path,
                record_path=record_path,
            )
        )
    return sorted(entries, key=lambda entry: entry.atom_name)


def _normalize_review_status(value: Any, trust_readiness: str, required_actions: Sequence[Any], blockers: Sequence[Any]) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "approved":
        return "approved"
    if trust_readiness in _READY_TRUST_STATES and not required_actions and not blockers:
        return "approved"
    if normalized in {"reviewed", "partial", "in_review", "reviewed_pending"}:
        return "reviewed_pending"
    return "missing"


def _normalize_review_verdict(value: Any, *, fallback_ready: bool = False) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"pass", "pass_with_limits"}:
        return normalized
    if normalized in _PASS_VERDICTS:
        return "pass"
    if fallback_ready:
        return "pass_with_limits"
    return "unknown"


def _normalize_overall_verdict(trust_readiness: str, semantic_verdict: str, blockers: Sequence[Any]) -> str:
    if blockers:
        return "limited_acceptability"
    if trust_readiness in _READY_TRUST_STATES or semantic_verdict in _PASS_VERDICTS:
        return "acceptable_with_limits"
    return "unknown"


def _normalize_trust_readiness(value: Any, required_actions: Sequence[Any], blockers: Sequence[Any]) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return "not_reviewed"
    if blockers or required_actions:
        if normalized in _READY_TRUST_STATES:
            return "reviewed_with_limits"
        return normalized
    if normalized in _READY_TRUST_STATES:
        return "reviewed_with_limits"
    return normalized


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if value is None:
        return []
    return [value]


def _row_list_or_bundle_list(row: dict[str, Any], bundle: dict[str, Any], key: str) -> list[Any]:
    if key in row:
        return _coerce_list(row.get(key))
    return _coerce_list(bundle.get(key))


def _row_atom_names(row: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key in ("atom_name", "atom_fqdn", "atom_key"):
        value = str(row.get(key) or "").strip()
        if value:
            names.append(value)
    for key in _coerce_list(row.get("atom_keys")):
        value = str(key or "").strip()
        if value:
            names.append(value)
    return sorted(dict.fromkeys(names))


def _canonical_atom_name(value: str) -> str:
    return value.split("@", 1)[0].strip()


def _flatten_row_bundle_entries(
    bundle: dict[str, Any],
    row: dict[str, Any],
    *,
    source_path: Path,
    record_path: str,
) -> list[ReviewBundleEntry]:
    identifiers = _row_atom_names(row)
    if not identifiers:
        raise ValueError(f"Row bundle entry in {source_path} is missing atom identifier fields")
    required_actions = _row_list_or_bundle_list(row, bundle, "required_actions")
    blockers = _row_list_or_bundle_list(row, bundle, "blocking_findings")
    trust_readiness = str(row.get("trust_readiness") or bundle.get("trust_readiness") or "").strip()
    semantic_verdict = str(
        row.get("semantic_verdict")
        or row.get("review_semantic_verdict")
        or bundle.get("semantic_verdict")
        or bundle.get("review_semantic_verdict")
        or ""
    ).strip()
    developer_semantic_verdict = str(
        row.get("developer_semantic_verdict")
        or row.get("review_developer_semantic_verdict")
        or row.get("review_developer_semantics_verdict")
        or bundle.get("developer_semantic_verdict")
        or bundle.get("review_developer_semantic_verdict")
        or bundle.get("review_developer_semantics_verdict")
        or ""
    ).strip()
    review_status = _normalize_review_status(
        row.get("review_status") or bundle.get("review_status"),
        trust_readiness,
        required_actions,
        blockers,
    )
    normalized_trust = _normalize_trust_readiness(trust_readiness, required_actions, blockers)
    patch = {
        "review_status": review_status,
        "review_priority": "review_now" if review_status == "approved" else "review_later",
        "review_semantic_verdict": _normalize_review_verdict(
            semantic_verdict,
            fallback_ready=review_status == "approved",
        ),
        "review_developer_semantics_verdict": _normalize_review_verdict(
            developer_semantic_verdict,
            fallback_ready=review_status == "approved",
        ),
        "trust_readiness": normalized_trust,
        "review_limitations": _row_list_or_bundle_list(row, bundle, "limitations"),
        "review_required_actions": required_actions,
        "trust_blockers": blockers,
        "authoritative_sources": _row_list_or_bundle_list(row, bundle, "authoritative_sources"),
        "review_record_path": str(row.get("review_record_path") or bundle.get("review_record_path") or record_path),
        "overall_verdict": _normalize_overall_verdict(normalized_trust, semantic_verdict, blockers),
    }
    if review_status == "approved":
        patch.update(
            {
                "structural_status": "pass",
                "semantic_status": "pass",
                "runtime_status": "pass",
                "developer_semantics_status": "pass",
            }
        )
    for key, value in row.items():
        if key in _IDENTIFIER_FIELDS or key in _ROW_CONTROL_FIELDS:
            continue
        if value is None:
            continue
        patch.setdefault(key, value)
    entries: list[ReviewBundleEntry] = []
    for identifier in identifiers:
        entries.append(
            ReviewBundleEntry(
                atom_name=_canonical_atom_name(identifier),
                atom_key=identifier if "." not in identifier.split("@", 1)[0] else None,
                patch=dict(patch),
                source_path=source_path,
                record_path=str(patch["review_record_path"]),
            )
        )
    return entries


def load_review_bundle_entries_from_workspace(base_dir: Path | None = None) -> list[ReviewBundleEntry]:
    """Load all review bundle entries from discovered provider repositories."""
    entries: list[ReviewBundleEntry] = []
    for path in discover_review_bundle_paths(base_dir):
        entries.extend(load_review_bundle_entries(path))
    return sorted(entries, key=lambda entry: (entry.atom_name, str(entry.source_path)))


def _merge_patch(target: dict[str, Any], patch: dict[str, Any], *, source_path: Path, record_path: str) -> None:
    for key, value in patch.items():
        if value is not None:
            target[key] = value
    if not target.get("review_record_path"):
        target["review_record_path"] = record_path


_STRUCTURAL_REFRESH_FIELDS = (
    "module_import_path",
    "module_path",
    "module_family",
    "domain_family",
    "wrapper_symbol",
    "argument_names",
    "argument_details",
    "uses_varargs",
    "uses_kwargs",
    "return_annotation",
    "docstring_summary",
    "has_docstring",
    "source_kind",
)


def _annotation_to_string(annotation: Any) -> str:
    if annotation is inspect.Signature.empty:
        return "Any"
    if isinstance(annotation, str):
        return annotation
    if getattr(annotation, "__module__", "") == "builtins" and getattr(annotation, "__name__", None):
        return str(annotation.__name__)
    return str(annotation).replace("typing.", "")


def _candidate_import_roots(source_path: Path) -> tuple[str, ...]:
    resolved = source_path.resolve()
    parts = resolved.parts
    candidates: list[str] = []
    for marker in ("src", "data", "docs"):
        if marker in parts:
            root = Path(*parts[: parts.index(marker)])
            src_root = root / "src"
            if src_root.is_dir():
                candidates.append(str(src_root))
            candidates.append(str(root))
            break
    return tuple(path for path in dict.fromkeys(candidates) if path)


@contextmanager
def _temporary_import_roots(paths: Sequence[str], *, module_name: str = ""):
    original = list(sys.path)
    additions = [path for path in paths if path and path not in sys.path]
    package_path_originals: dict[str, list[str]] = {}
    if additions:
        sys.path[:0] = additions
    if module_name:
        parts = module_name.split(".")
        for depth in range(1, len(parts)):
            package_name = ".".join(parts[:depth])
            package = sys.modules.get(package_name)
            package_path = getattr(package, "__path__", None)
            if package_path is None:
                continue
            original_paths = list(package_path)
            extras: list[str] = []
            for root in additions:
                candidate = Path(root, *parts[:depth])
                candidate_str = str(candidate)
                if candidate.is_dir() and candidate_str not in package_path:
                    extras.append(candidate_str)
            if extras:
                package_path_originals[package_name] = original_paths
                package_path[:0] = extras
    importlib.invalidate_caches()
    try:
        yield
    finally:
        for package_name, original_paths in package_path_originals.items():
            package = sys.modules.get(package_name)
            package_path = getattr(package, "__path__", None)
            if package_path is not None:
                package_path[:] = original_paths
        sys.path[:] = original
        importlib.invalidate_caches()


def _base_entry_from_callable(atom_name: str, *, import_roots: Sequence[str] = ()) -> dict[str, Any]:
    module_name, _, symbol_name = atom_name.rpartition(".")
    if not module_name or not symbol_name:
        raise ValueError(f"Invalid atom name {atom_name!r}")
    with _temporary_import_roots(import_roots, module_name=module_name):
        module = importlib.import_module(module_name)
    target = getattr(module, symbol_name)
    unwrapped = inspect.unwrap(target)
    signature = inspect.signature(target)
    argument_details: list[dict[str, Any]] = []
    argument_names: list[str] = []
    uses_varargs = False
    uses_kwargs = False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            uses_varargs = True
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            uses_kwargs = True
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            annotation = _annotation_to_string(parameter.annotation)
            argument_names.append(parameter.name)
            argument_details.append(
                {
                    "name": parameter.name,
                    "annotation": annotation,
                    "required": parameter.default is inspect.Signature.empty,
                    "kind": parameter.kind.name.lower(),
                }
            )
    doc = inspect.getdoc(unwrapped) or inspect.getdoc(target) or ""
    source_file = inspect.getsourcefile(unwrapped) or inspect.getsourcefile(target) or ""
    wrapper_symbol = getattr(unwrapped, "__name__", getattr(target, "__name__", symbol_name))
    return {
        "atom_name": atom_name,
        "atom_key": atom_name,
        "atom_id": atom_name,
        "module_import_path": module_name,
        "module_path": source_file,
        "module_family": module_name.split(".")[-2] if "." in module_name else module_name,
        "domain_family": ".".join(module_name.split(".")[2:-1]) if module_name.startswith("sciona.") else module_name,
        "wrapper_symbol": wrapper_symbol,
        "argument_names": argument_names,
        "argument_details": argument_details,
        "uses_varargs": uses_varargs,
        "uses_kwargs": uses_kwargs,
        "return_annotation": _annotation_to_string(signature.return_annotation),
        "docstring_summary": (doc.splitlines()[0].strip() if doc else ""),
        "has_docstring": bool(doc),
        "authoritative_sources": [],
        "source_kind": "hand_written",
        "review_status": "missing",
        "review_priority": "review_later",
        "review_semantic_verdict": "unknown",
        "review_developer_semantics_verdict": "unknown",
        "trust_readiness": "not_reviewed",
        "review_required_actions": [],
        "review_limitations": [],
        "trust_blockers": [],
        "structural_status": "unknown",
        "semantic_status": "unknown",
        "runtime_status": "unknown",
        "developer_semantics_status": "unknown",
        "overall_verdict": "acceptable_with_limits",
        "risk_tier": "low",
        "risk_score": 0,
        "risk_dimensions": {},
        "risk_reasons": [],
        "acceptability_score": 70,
        "acceptability_band": "acceptable_with_limits_candidate",
        "parity_coverage_level": "unknown",
        "parity_test_status": "unknown",
        "parity_fixture_count": 0,
        "parity_case_count": 0,
        "required_actions": [],
        "blocking_findings": [],
    }


def _refresh_structural_fields_from_callable(target: dict[str, Any], atom_name: str, *, import_roots: Sequence[str] = ()) -> None:
    live_entry = _base_entry_from_callable(atom_name, import_roots=import_roots)
    for field in _STRUCTURAL_REFRESH_FIELDS:
        target[field] = live_entry[field]


def merge_audit_manifest_entries(
    manifest_entries: Sequence[dict[str, Any]],
    review_entries: Sequence[ReviewBundleEntry],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return manifest entries with review bundle patches applied."""
    merged_by_name = {
        str(entry.get("atom_name") or ""): dict(entry)
        for entry in manifest_entries
        if str(entry.get("atom_name") or "").strip()
    }
    merged_by_key = {
        str(entry.get("atom_key") or ""): merged_by_name[str(entry.get("atom_name") or "")]
        for entry in manifest_entries
        if str(entry.get("atom_name") or "").strip() and str(entry.get("atom_key") or "").strip()
    }
    order = [str(entry.get("atom_name") or "").strip() for entry in manifest_entries if str(entry.get("atom_name") or "").strip()]

    skipped: list[str] = []
    for entry in review_entries:
        import_roots = _candidate_import_roots(entry.source_path)
        current = merged_by_name.get(entry.atom_name)
        if current is None and entry.atom_key:
            current = merged_by_key.get(entry.atom_key)
        if current is None:
            try:
                current = _base_entry_from_callable(entry.atom_name, import_roots=import_roots)
            except Exception:
                skipped.append(entry.atom_name)
                continue
            if entry.atom_key:
                current["atom_key"] = entry.atom_key
            merged_by_name[entry.atom_name] = current
            order.append(entry.atom_name)
        else:
            merged_by_name[str(current.get("atom_name") or entry.atom_name)] = current
            key = str(current.get("atom_key") or entry.atom_key or "").strip()
            if key:
                merged_by_key[key] = current
            try:
                _refresh_structural_fields_from_callable(current, entry.atom_name, import_roots=import_roots)
            except Exception:
                pass
        _merge_patch(current, entry.patch, source_path=entry.source_path, record_path=entry.record_path)

    return [merged_by_name[name] for name in sorted(dict.fromkeys(order))], sorted(dict.fromkeys(skipped))


def load_audit_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load the manifest container object."""
    manifest_path = path or discover_audit_manifest_path()
    payload = _load_json_object(manifest_path)
    atoms = payload.get("atoms", [])
    if not isinstance(atoms, list):
        raise ValueError(f"Expected manifest atoms list in {manifest_path}")
    payload["atoms"] = [entry for entry in atoms if isinstance(entry, dict)]
    return payload


def merge_audit_manifest_with_review_bundles(
    *,
    manifest_path: Path | None = None,
    base_dir: Path | None = None,
    review_bundle_paths: Iterable[Path] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply provider review bundle files to the audit manifest."""
    resolved_manifest_path = manifest_path or discover_audit_manifest_path(base_dir)
    manifest = load_audit_manifest(resolved_manifest_path)
    bundle_paths = tuple(review_bundle_paths) if review_bundle_paths is not None else discover_review_bundle_paths(base_dir)
    review_entries: list[ReviewBundleEntry] = []
    for path in bundle_paths:
        review_entries.extend(load_review_bundle_entries(path))
    review_entries = sorted(review_entries, key=lambda entry: (entry.atom_name, str(entry.source_path)))
    merged_atoms, skipped_atom_names = merge_audit_manifest_entries(manifest["atoms"], review_entries)
    existing_names = {str(entry.get("atom_name") or "").strip() for entry in manifest["atoms"] if str(entry.get("atom_name") or "").strip()}
    merged_names = [str(entry.get("atom_name") or "").strip() for entry in merged_atoms if str(entry.get("atom_name") or "").strip()]

    summary = {
        "manifest_path": str(resolved_manifest_path),
        "bundle_paths": [str(path) for path in bundle_paths],
        "bundle_entry_count": len(review_entries),
        "manifest_entry_count": len(manifest["atoms"]),
        "merged_entry_count": len(merged_atoms),
        "created_entry_count": sum(1 for name in merged_names if name not in existing_names),
        "updated_entry_count": sum(1 for name in merged_names if name in existing_names),
        "skipped_unresolved_atom_count": len(skipped_atom_names),
        "skipped_unresolved_atoms": skipped_atom_names,
        "dry_run": dry_run,
    }

    manifest["atoms"] = merged_atoms
    manifest.setdefault("metadata", {})
    if isinstance(manifest["metadata"], dict):
        manifest["metadata"]["generated_at"] = datetime.now(timezone.utc).isoformat()
        manifest["metadata"]["generator"] = "scripts/apply_audit_review_bundles.py"
        manifest["metadata"]["phase"] = "phase_1_review_bundle_merge"

    if not dry_run:
        resolved_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge provider-owned audit review bundles into audit_manifest.json")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    summary = merge_audit_manifest_with_review_bundles(
        manifest_path=args.manifest_path,
        base_dir=args.base_dir,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge provider audit review bundles into the audit manifest.")
    parser.add_argument("--manifest", type=Path, default=None, help="Path to data/audit_manifest.json")
    parser.add_argument("--base-dir", type=Path, default=None, help="Workspace root containing sibling provider repos")
    parser.add_argument("--dry-run", action="store_true", help="Do not write the manifest, only report the summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = merge_audit_manifest_with_review_bundles(
        manifest_path=args.manifest,
        base_dir=args.base_dir,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
