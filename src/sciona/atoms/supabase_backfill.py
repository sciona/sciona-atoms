"""Provider-owned Supabase backfill helpers for file-backed atom metadata."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from sciona.atoms.provider_inventory import (
    discover_audit_manifest_path,
    discover_references_registry_path,
    iter_provider_artifact_files,
    namespace_prefix_for_artifact_root,
)

if TYPE_CHECKING:
    from supabase import Client

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 50
DEFAULT_RUNNER_VERSION = "backfill-v1"
DEFAULT_DEJARGON_LANGUAGE = "en"

MATCH_TYPE_TO_SOURCE = {
    "manual": "manual",
    "ast_subgraph": "llm_extracted",
    "name_heuristic": "llm_extracted",
}
VALID_LEVELS = {
    "kernel_proof",
    "type_checked",
    "contract_checked",
    "unverified",
}
VALID_ACCEPTABILITY_BANDS = {
    "unknown",
    "acceptable_with_limits",
    "acceptable_with_limits_candidate",
    "limited_acceptability",
}


def create_supabase_client_from_env() -> Any:
    """Create a service-role Supabase client from environment variables."""
    from supabase import create_client

    service_key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_SERVICE_KEY")
        or os.environ.get("SCIONA_SUPABASE_SERVICE_ROLE_KEY")
        or os.environ["SCIONA_SUPABASE_SERVICE_KEY"]
    )
    url = os.environ.get("SUPABASE_URL") or os.environ["SCIONA_SUPABASE_URL"]
    return create_client(url, service_key)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _artifact_roots(single_root: Path | None) -> tuple[Path, ...]:
    if single_root is not None:
        return (single_root.expanduser().resolve(),)
    roots = {
        path.parent.resolve()
        for path in iter_provider_artifact_files("matches.json")
    }
    roots.update(
        path.parent.resolve()
        for path in iter_provider_artifact_files("references.json")
    )
    for path in iter_provider_artifact_files("cdg.json"):
        roots.add(path.parent.resolve())
    for path in iter_provider_artifact_files("*_cdg.json"):
        roots.add(path.parent.resolve())
    if roots:
        return tuple(sorted(roots))
    return tuple()


def iter_metadata_files(
    filename: str,
    *,
    single_root: Path | None = None,
) -> list[Path]:
    """Return metadata files from one root or all discovered providers."""
    roots = _artifact_roots(single_root) if single_root is not None else None
    return iter_provider_artifact_files(filename, roots=roots)


def namespace_from_path(file_path: Path) -> str:
    """Derive the dotted namespace from an artifact file path."""
    raw_parent = file_path.parent
    parents = raw_parent.resolve() if file_path.is_absolute() else raw_parent
    for ancestor in (parents, *parents.parents):
        prefix = namespace_prefix_for_artifact_root(ancestor)
        if prefix != (ancestor.name,):
            relative = parents.relative_to(ancestor)
            parts: list[str] = []
            for part in relative.parts:
                if part == "_artifacts":
                    break
                parts.append(part)
            return ".".join((*prefix, *parts))

    parts = []
    for part in parents.parts:
        if part == "_artifacts":
            break
        parts.append(part)
    return ".".join(parts)


def resolve_atom_id(supabase: "Client", namespace: str, short_name: str) -> str | None:
    """Resolve an atom_id by exact FQDN first, then suffix fallback."""
    fqdn = f"{namespace}.{short_name}"
    response = (
        supabase.table("atoms")
        .select("atom_id")
        .eq("fqdn", fqdn)
        .limit(1)
        .execute()
    )
    if response.data:
        return response.data[0]["atom_id"]

    response = (
        supabase.table("atoms")
        .select("atom_id")
        .like("fqdn", f"%.{short_name}")
        .limit(1)
        .execute()
    )
    if response.data:
        return response.data[0]["atom_id"]
    return None


def fetch_atom_lookup(supabase: "Client") -> dict[str, str]:
    """Fetch the current ``fqdn -> atom_id`` lookup from Supabase."""
    response = supabase.table("atoms").select("atom_id, fqdn").execute()
    return {row["fqdn"]: row["atom_id"] for row in response.data or []}


def load_manifest_entries(path: Path | None = None) -> list[dict[str, Any]]:
    """Load audit manifest entries."""
    manifest_path = path or discover_audit_manifest_path()
    payload = _load_json(manifest_path)
    atoms = payload.get("atoms", [])
    if not isinstance(atoms, list):
        raise ValueError(f"Expected manifest atoms list in {manifest_path}")
    return atoms


def load_manifest_argument_names(path: Path | None = None) -> dict[str, list[str]]:
    """Return ``atom_name -> argument_names`` from the audit manifest."""
    return {
        entry["atom_name"]: list(entry.get("argument_names", []))
        for entry in load_manifest_entries(path)
        if "atom_name" in entry
    }


def derive_atom_fqdn(cdg_path: Path, atoms_root: Path, node_name: str) -> str:
    """Derive a fully qualified atom name from a CDG file path."""
    rel_parts = cdg_path.parent.relative_to(atoms_root).parts
    namespace_prefix = namespace_prefix_for_artifact_root(atoms_root)
    return ".".join((*namespace_prefix, *rel_parts, node_name))


def build_io_spec_rows(atom_id: str, node: dict[str, Any]) -> list[dict[str, Any]]:
    """Map a single CDG atomic node to input/output rows."""
    rows: list[dict[str, Any]] = []
    for ordinal, spec in enumerate(node.get("inputs", [])):
        rows.append(
            {
                "atom_id": atom_id,
                "version_id": None,
                "direction": "input",
                "name": spec["name"],
                "type_desc": spec.get("type_desc") or "Any",
                "constraints": spec.get("constraints") or "",
                "required": True,
                "default_value_repr": "",
                "ordinal": ordinal,
            }
        )
    for ordinal, spec in enumerate(node.get("outputs", [])):
        rows.append(
            {
                "atom_id": atom_id,
                "version_id": None,
                "direction": "output",
                "name": spec["name"],
                "type_desc": spec.get("type_desc") or "Any",
                "constraints": spec.get("constraints") or "",
                "required": True,
                "default_value_repr": "",
                "ordinal": ordinal,
            }
        )
    return rows


def input_name_mismatch(cdg_input_names: list[str], manifest_arg_names: list[str]) -> bool:
    """Return whether cross-validation should warn."""
    return bool(manifest_arg_names) and cdg_input_names != manifest_arg_names


def _iter_cdg_files(single_root: Path | None = None) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    roots = _artifact_roots(single_root) if single_root is not None else None
    for filename in ("cdg.json", "*_cdg.json"):
        for path in iter_provider_artifact_files(filename, roots=roots):
            best_root: Path | None = None
            for parent in path.parents:
                prefix = namespace_prefix_for_artifact_root(parent)
                if prefix != (parent.name,):
                    best_root = parent.resolve()
                    break
            if best_root is not None:
                pairs.append((best_root, path.resolve()))
    deduped = {(root, path) for root, path in pairs}
    return sorted(deduped, key=lambda item: str(item[1]))


def backfill_io_specs(
    supabase: Any,
    *,
    atoms_root: Path | None = None,
    audit_manifest_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Populate ``atom_io_specs`` from provider CDG files."""
    atom_lookup = fetch_atom_lookup(supabase)
    manifest_args = load_manifest_argument_names(audit_manifest_path)
    stats = {"inserted": 0, "skipped_no_atom": 0, "cdg_files": 0, "cross_val_warnings": 0}

    for artifact_root, cdg_path in _iter_cdg_files(atoms_root):
        stats["cdg_files"] += 1
        cdg = _load_json(cdg_path)
        for node in cdg.get("nodes", []):
            if node.get("status") != "atomic":
                continue

            node_name = str(node.get("name", ""))
            atom_fqdn = derive_atom_fqdn(cdg_path, artifact_root, node_name)
            atom_id = atom_lookup.get(atom_fqdn)
            if not atom_id:
                logger.warning("No atom found for %s (CDG %s)", atom_fqdn, cdg_path)
                stats["skipped_no_atom"] += 1
                continue

            cdg_input_names = [spec["name"] for spec in node.get("inputs", [])]
            manifest_arg_names = manifest_args.get(atom_fqdn, [])
            if input_name_mismatch(cdg_input_names, manifest_arg_names):
                logger.warning(
                    "Input name mismatch for %s: CDG=%s manifest=%s",
                    atom_fqdn,
                    cdg_input_names,
                    manifest_arg_names,
                )
                stats["cross_val_warnings"] += 1

            rows = build_io_spec_rows(atom_id, node)
            if not rows:
                continue
            if dry_run:
                stats["inserted"] += len(rows)
                continue

            (
                supabase.table("atom_io_specs")
                .delete()
                .eq("atom_id", atom_id)
                .is_("version_id", "null")
                .execute()
            )
            supabase.table("atom_io_specs").insert(rows).execute()
            stats["inserted"] += len(rows)

    return stats


def build_parameter_rows(atom_id: str, atom_entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Map manifest argument details into ``atom_parameters`` rows."""
    rows: list[dict[str, Any]] = []
    for position, arg in enumerate(atom_entry.get("argument_details", [])):
        rows.append(
            {
                "atom_id": atom_id,
                "version_id": None,
                "name": arg["name"],
                "position": position,
                "kind": arg.get("kind", "positional_or_keyword"),
                "type_desc": arg.get("annotation") or "Any",
                "required": arg.get("required", True),
                "default_value_repr": "",
                "technical_description": "",
                "dejargonized_description": "",
                "constraints_json": {},
            }
        )

    next_position = len(rows)
    if atom_entry.get("uses_varargs"):
        rows.append(
            {
                "atom_id": atom_id,
                "version_id": None,
                "name": "*args",
                "position": next_position,
                "kind": "varargs",
                "type_desc": "Any",
                "required": False,
                "default_value_repr": "",
                "technical_description": "",
                "dejargonized_description": "",
                "constraints_json": {},
            }
        )
        next_position += 1
    if atom_entry.get("uses_kwargs"):
        rows.append(
            {
                "atom_id": atom_id,
                "version_id": None,
                "name": "**kwargs",
                "position": next_position,
                "kind": "kwargs",
                "type_desc": "Any",
                "required": False,
                "default_value_repr": "",
                "technical_description": "",
                "dejargonized_description": "",
                "constraints_json": {},
            }
        )
    return rows


def backfill_parameters(
    supabase: Any,
    *,
    audit_manifest_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Populate ``atom_parameters`` from the audit manifest."""
    manifest = load_manifest_entries(audit_manifest_path)
    atom_lookup = fetch_atom_lookup(supabase)
    stats = {"inserted": 0, "skipped_no_atom": 0, "atoms_processed": 0}

    for atom_entry in manifest:
        fqdn = atom_entry["atom_name"]
        atom_id = atom_lookup.get(fqdn)
        if not atom_id:
            stats["skipped_no_atom"] += 1
            continue
        stats["atoms_processed"] += 1
        rows = build_parameter_rows(atom_id, atom_entry)
        if not rows:
            continue
        if dry_run:
            stats["inserted"] += len(rows)
            continue

        (
            supabase.table("atom_parameters")
            .delete()
            .eq("atom_id", atom_id)
            .is_("version_id", "null")
            .execute()
        )
        supabase.table("atom_parameters").insert(rows).execute()
        stats["inserted"] += len(rows)

    return stats


def choose_technical_content(atom_entry: dict[str, Any], atom_row: dict[str, Any]) -> str:
    """Prefer manifest docstring summary over the atoms.description fallback."""
    return str(atom_entry.get("docstring_summary") or atom_row.get("description") or "").strip()


def build_technical_description_row(atom_id: str, content: str) -> dict[str, Any]:
    """Build a technical description row."""
    return {
        "atom_id": atom_id,
        "kind": "technical",
        "language": DEFAULT_DEJARGON_LANGUAGE,
        "content": content,
        "generated_by": "backfill-v1",
        "reviewed": False,
        "jargon_score": 1.0,
    }


def dedupe_technical_description_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse duplicate technical-description upserts by conflict key."""
    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["atom_id"]),
            str(row["kind"]),
            str(row["language"]),
        )
        candidate = dict(row)
        incumbent = deduped.get(key)
        if incumbent is None or _technical_description_rank(candidate) > _technical_description_rank(
            incumbent
        ):
            deduped[key] = candidate
    return [deduped[key] for key in sorted(deduped)]


def backfill_technical_descriptions(
    supabase: Any,
    *,
    audit_manifest_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Populate technical ``atom_descriptions`` rows from the audit manifest."""
    manifest = load_manifest_entries(audit_manifest_path)
    atoms_resp = supabase.table("atoms").select("atom_id, fqdn, description").execute()
    atom_lookup = {row["fqdn"]: row for row in atoms_resp.data or []}

    stats = {"inserted": 0, "skipped_no_content": 0, "skipped_no_atom": 0}
    rows: list[dict[str, Any]] = []
    for atom_entry in manifest:
        fqdn = atom_entry["atom_name"]
        atom_row = atom_lookup.get(fqdn)
        if not atom_row:
            stats["skipped_no_atom"] += 1
            continue
        content = choose_technical_content(atom_entry, atom_row)
        if not content:
            stats["skipped_no_content"] += 1
            continue
        rows.append(build_technical_description_row(atom_row["atom_id"], content))
    rows = dedupe_technical_description_rows(rows)

    for start in range(0, len(rows), 100):
        batch = rows[start : start + 100]
        if not dry_run:
            supabase.table("atom_descriptions").upsert(batch, on_conflict="atom_id,kind,language").execute()
        stats["inserted"] += len(batch)
    return stats


def load_registry(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the canonical bibliography keyed by ``ref_id``."""
    registry_path = path or discover_references_registry_path()
    data = _load_json(registry_path)
    refs = data.get("references", data)
    if not isinstance(refs, dict):
        raise ValueError(f"Registry payload at {registry_path} must contain an object")
    return {str(ref_id): dict(entry) for ref_id, entry in refs.items()}


def build_registry_row(ref_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Map a registry JSON entry to ``references_registry`` columns."""
    return {
        "ref_id": ref_id,
        "ref_type": entry.get("type", "paper"),
        "title": entry.get("title", ""),
        "authors": entry.get("authors", []),
        "year": entry.get("year"),
        "venue": entry.get("venue", ""),
        "doi": entry.get("doi"),
        "url": entry.get("url", ""),
        "bibtex_key": entry.get("bibtex_key", ref_id),
        "bibtex_raw": entry.get("bibtex_raw", ""),
    }


def backfill_references_registry(
    supabase: Any,
    *,
    registry_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Populate ``references_registry`` from the canonical JSON registry."""
    registry = load_registry(registry_path)
    stats = {"upserted": 0, "errors": 0}
    for ref_id, entry in sorted(registry.items()):
        row = build_registry_row(ref_id, entry)
        if dry_run:
            stats["upserted"] += 1
            continue
        try:
            supabase.table("references_registry").upsert(row, on_conflict="ref_id").execute()
            stats["upserted"] += 1
        except Exception:
            logger.exception("Failed to upsert registry entry %s", ref_id)
            stats["errors"] += 1
    return stats


def extract_fqdn(atom_key: str) -> str:
    """Extract the FQDN from a manifest-key reference entry."""
    fqdn, _, _rest = atom_key.partition("@")
    return fqdn


def build_ref_key(ref_id: str, registry_entry: dict[str, Any]) -> str:
    """Resolve the ``atom_references`` deduplication key."""
    doi = registry_entry.get("doi")
    if doi:
        return str(doi)
    if ref_id:
        return ref_id
    title = str(registry_entry.get("title", "unknown"))
    return title[:80]


def map_source(match_metadata: dict[str, Any]) -> str:
    """Map per-atom match type into the ``atom_references.source`` enum."""
    return MATCH_TYPE_TO_SOURCE.get(str(match_metadata.get("match_type", "")), "manual")


def build_atom_reference_row(
    atom_id: str,
    ref_id: str,
    registry_entry: dict[str, Any],
    match_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Map an atom/reference binding into ``atom_references`` columns."""
    return {
        "atom_id": atom_id,
        "ref_id": ref_id,
        "ref_key": build_ref_key(ref_id, registry_entry),
        "doi": registry_entry.get("doi"),
        "title": registry_entry.get("title", ""),
        "authors": registry_entry.get("authors", []),
        "year": registry_entry.get("year"),
        "url": registry_entry.get("url", ""),
        "relevance_note": match_metadata.get("notes", ""),
        "confidence": match_metadata.get("confidence", ""),
        "matched_nodes": match_metadata.get("matched_nodes", []),
        "source": map_source(match_metadata),
        "verified": False,
    }


def iter_reference_files(atoms_root: Path | Sequence[Path] | None) -> list[Path]:
    """List ``references.json`` files from one or many artifact roots."""
    if atoms_root is None:
        return iter_provider_artifact_files("references.json")
    if isinstance(atoms_root, Path):
        roots: Sequence[Path] = (atoms_root,)
    else:
        roots = tuple(atoms_root)

    files: list[Path] = []
    for root in roots:
        files.extend(
            path
            for path in root.rglob("references.json")
            if "__pycache__" not in path.parts
        )
    return sorted({path.resolve() for path in files})


def backfill_references(
    supabase: Any,
    *,
    atoms_root: Path | Sequence[Path] | None = None,
    registry_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Populate ``atom_references`` from per-atom ``references.json`` files."""
    registry = load_registry(registry_path)
    atom_lookup = fetch_atom_lookup(supabase)
    stats = {"inserted": 0, "skipped_no_atom": 0, "skipped_no_registry": 0, "errors": 0}

    for refs_path in iter_reference_files(atoms_root):
        data = _load_json(refs_path)
        atoms_block = data.get("atoms", {})
        if not isinstance(atoms_block, dict):
            logger.warning("Skipping malformed atoms block in %s", refs_path)
            stats["errors"] += 1
            continue

        for atom_key, atom_data in atoms_block.items():
            fqdn = extract_fqdn(atom_key)
            atom_id = atom_lookup.get(fqdn)
            if not atom_id:
                logger.warning("No atom found for %s (from %s)", fqdn, refs_path)
                stats["skipped_no_atom"] += 1
                continue

            references = atom_data.get("references", [])
            if not isinstance(references, list):
                logger.warning("Skipping malformed references list for %s in %s", fqdn, refs_path)
                stats["errors"] += 1
                continue

            for ref_binding in references:
                ref_id = str(ref_binding.get("ref_id", "")).strip()
                if not ref_id:
                    continue
                registry_entry = registry.get(ref_id)
                if not registry_entry:
                    stats["skipped_no_registry"] += 1
                    continue
                row = build_atom_reference_row(
                    atom_id,
                    ref_id,
                    registry_entry,
                    dict(ref_binding.get("match_metadata", {})),
                )
                if dry_run:
                    stats["inserted"] += 1
                    continue
                try:
                    (
                        supabase.table("atom_references")
                        .upsert(row, on_conflict="atom_id,ref_key")
                        .execute()
                    )
                    stats["inserted"] += 1
                except Exception:
                    logger.exception("Failed to upsert ref %s for atom %s", ref_id, fqdn)
                    stats["errors"] += 1
    return stats


def build_rollup_row(atom_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Map an audit manifest entry to a single ``atom_audit_rollups`` row."""
    return {
        "atom_id": atom_id,
        "overall_verdict": entry.get("overall_verdict") or "unknown",
        "structural_status": entry.get("structural_status") or "unknown",
        "runtime_status": entry.get("runtime_status") or "unknown",
        "semantic_status": entry.get("semantic_status") or "unknown",
        "developer_semantics_status": entry.get("developer_semantics_status") or "unknown",
        "risk_tier": entry.get("risk_tier") or "medium",
        "risk_score": entry.get("risk_score", 0),
        "risk_dimensions": entry.get("risk_dimensions") or {},
        "risk_reasons": entry.get("risk_reasons") or [],
        "acceptability_score": entry.get("acceptability_score", 0),
        "acceptability_band": normalize_acceptability_band(
            entry.get("acceptability_band"),
            acceptability_score=entry.get("acceptability_score"),
        ),
        "parity_coverage_level": entry.get("parity_coverage_level") or "unknown",
        "parity_test_status": entry.get("parity_test_status") or "unknown",
        "parity_fixture_count": entry.get("parity_fixture_count", 0),
        "parity_case_count": entry.get("parity_case_count", 0),
        "review_status": entry.get("review_status") or "missing",
        "review_semantic_verdict": entry.get("review_semantic_verdict") or "unknown",
        "review_developer_semantics_verdict": entry.get("review_developer_semantics_verdict") or "unknown",
        "review_limitations": entry.get("review_limitations") or [],
        "review_required_actions": entry.get("review_required_actions") or [],
        "trust_readiness": entry.get("trust_readiness") or "not_ready",
        "trust_blockers": entry.get("trust_blockers") or [],
    }


def normalize_acceptability_band(
    value: Any,
    *,
    acceptability_score: Any = None,
) -> str:
    """Collapse newer manifest acceptability bands into the current DB taxonomy."""
    normalized = str(value or "unknown").strip()
    if normalized in VALID_ACCEPTABILITY_BANDS:
        return normalized

    score: int | None = None
    if acceptability_score is not None:
        try:
            score = int(acceptability_score)
        except (TypeError, ValueError):
            score = None

    if normalized == "review_ready":
        return "acceptable_with_limits"
    if normalized == "acceptable_with_limits_candidate":
        return "acceptable_with_limits_candidate"
    if normalized == "limited_acceptability":
        return "limited_acceptability"
    if normalized in {"misleading_candidate", "broken_candidate"}:
        return "unknown"

    if score is None:
        return "unknown"
    if score >= 85:
        return "acceptable_with_limits"
    if score >= 70:
        return "acceptable_with_limits_candidate"
    if score >= 50:
        return "limited_acceptability"
    return "unknown"


def _technical_description_rank(row: dict[str, Any]) -> tuple[int, int, str]:
    content = str(row.get("content") or "").strip()
    return (1 if content else 0, len(content), content)


def _stable_row_json(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


_OVERALL_VERDICT_RANK = {
    "trusted": 5,
    "acceptable_with_limits": 4,
    "limited_acceptability": 3,
    "misleading": 2,
    "broken": 1,
    "unknown": 0,
}


def _audit_rollup_rank(row: dict[str, Any]) -> tuple[int, int, int, int, str]:
    overall = str(row.get("overall_verdict") or "unknown")
    return (
        int(row.get("acceptability_score") or 0),
        -int(row.get("risk_score") or 0),
        int(row.get("parity_case_count") or 0) + int(row.get("parity_fixture_count") or 0),
        _OVERALL_VERDICT_RANK.get(overall, -1),
        json.dumps(row, sort_keys=True),
    )


def dedupe_audit_rollup_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse duplicate audit-rollup upserts by ``atom_id``."""
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["atom_id"])
        candidate = dict(row)
        incumbent = deduped.get(key)
        if incumbent is None or _audit_rollup_rank(candidate) > _audit_rollup_rank(incumbent):
            deduped[key] = candidate
    return [deduped[key] for key in sorted(deduped)]


def dedupe_uncertainty_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse exact duplicate uncertainty rows deterministically."""
    deduped = {_stable_row_json(dict(row)): dict(row) for row in rows}
    return [deduped[key] for key in sorted(deduped)]


def dedupe_verification_match_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse exact duplicate verification rows deterministically."""
    deduped = {_stable_row_json(dict(row)): dict(row) for row in rows}
    return [deduped[key] for key in sorted(deduped)]


def backfill_audit_rollups(
    supabase: Any,
    *,
    manifest_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
) -> dict[str, int]:
    """Populate ``atom_audit_rollups`` from the audit manifest."""
    manifest_atoms = load_manifest_entries(manifest_path)
    atom_lookup = fetch_atom_lookup(supabase)
    stats = {"manifest_atoms": len(manifest_atoms), "upserted": 0, "skipped_no_atom": 0}
    batch: list[dict[str, Any]] = []

    for entry in manifest_atoms:
        fqdn = str(entry.get("atom_name", "") or "")
        atom_id = atom_lookup.get(fqdn)
        if atom_id is None:
            stats["skipped_no_atom"] += 1
            continue

        batch.append(build_rollup_row(atom_id, entry))
        if len(batch) >= batch_size:
            batch = dedupe_audit_rollup_rows(batch)
            if not dry_run:
                supabase.table("atom_audit_rollups").upsert(batch, on_conflict="atom_id").execute()
            stats["upserted"] += len(batch)
            batch = []

    if batch:
        batch = dedupe_audit_rollup_rows(batch)
        if not dry_run:
            supabase.table("atom_audit_rollups").upsert(batch, on_conflict="atom_id").execute()
        stats["upserted"] += len(batch)
    return stats


def build_evidence_rows(
    atom_id: str,
    entry: dict[str, Any],
    *,
    runner_version: str = DEFAULT_RUNNER_VERSION,
) -> list[dict[str, Any]]:
    """Build synthetic audit evidence rows for one manifest atom entry."""
    rows: list[dict[str, Any]] = []
    common = {
        "atom_id": atom_id,
        "source_kind": "automated",
        "runner_version": runner_version,
        "source_revision": entry.get("source_revision") or "",
        "upstream_version": entry.get("upstream_version") or "",
    }

    structural = entry.get("structural_status")
    if structural is not None:
        rows.append(
            {
                **common,
                "audit_type": "structural_audit",
                "passed": structural == "pass",
                "details": {
                    "status": structural,
                    "findings": entry.get("structural_findings", []),
                    "finding_details": entry.get("structural_finding_details", []),
                },
            }
        )

    semantic = entry.get("semantic_status")
    if semantic not in (None, "unknown"):
        rows.append(
            {
                **common,
                "audit_type": "semantic_audit",
                "passed": semantic == "pass",
                "details": {
                    "status": semantic,
                    "findings": entry.get("semantic_findings", []),
                    "finding_details": entry.get("semantic_finding_details", []),
                },
            }
        )

    risk_tier = entry.get("risk_tier")
    if risk_tier is not None:
        rows.append(
            {
                **common,
                "audit_type": "risk_assessment",
                "passed": risk_tier == "low",
                "details": {
                    "risk_tier": risk_tier,
                    "risk_score": entry.get("risk_score", 0),
                    "risk_dimensions": entry.get("risk_dimensions", {}),
                    "risk_reasons": entry.get("risk_reasons", []),
                },
            }
        )

    parity = entry.get("parity_coverage_level")
    if parity not in (None, "unknown", "none"):
        rows.append(
            {
                **common,
                "audit_type": "parity_check",
                "passed": parity in ("positive_and_negative", "parity_or_usage_equivalent"),
                "details": {
                    "coverage_level": parity,
                    "coverage_reasons": entry.get("parity_coverage_reasons", []),
                    "test_status": entry.get("parity_test_status", "unknown"),
                    "fixture_count": entry.get("parity_fixture_count", 0),
                    "case_count": entry.get("parity_case_count", 0),
                    "usage_test_coverage": entry.get("usage_test_coverage", ""),
                },
            }
        )

    runtime = entry.get("runtime_status")
    if runtime not in (None, "not_applicable"):
        rows.append(
            {
                **common,
                "audit_type": "smoke_test",
                "passed": runtime == "pass",
                "details": {
                    "status": runtime,
                    "status_basis": (entry.get("status_basis") or {}).get("runtime", []),
                },
            }
        )

    return rows


def _fetch_existing_evidence_keys(
    supabase: Any,
    *,
    runner_version: str,
) -> set[tuple[str, str]]:
    response = (
        supabase.table("atom_audit_evidence")
        .select("atom_id, audit_type")
        .eq("runner_version", runner_version)
        .execute()
    )
    return {
        (row["atom_id"], row["audit_type"])
        for row in (response.data or [])
        if row.get("atom_id") and row.get("audit_type")
    }


def backfill_audit_evidence(
    supabase: Any,
    *,
    manifest_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    runner_version: str = DEFAULT_RUNNER_VERSION,
    dry_run: bool = False,
) -> dict[str, int]:
    """Populate ``atom_audit_evidence`` from the audit manifest."""
    manifest_atoms = load_manifest_entries(manifest_path)
    atom_lookup = fetch_atom_lookup(supabase)
    existing_keys = _fetch_existing_evidence_keys(supabase, runner_version=runner_version)

    stats = {
        "manifest_atoms": len(manifest_atoms),
        "inserted": 0,
        "skipped_no_atom": 0,
        "skipped_existing": 0,
    }
    batch: list[dict[str, Any]] = []
    for entry in manifest_atoms:
        fqdn = str(entry.get("atom_name", "") or "")
        atom_id = atom_lookup.get(fqdn)
        if atom_id is None:
            stats["skipped_no_atom"] += 1
            continue

        for row in build_evidence_rows(atom_id, entry, runner_version=runner_version):
            key = (row["atom_id"], row["audit_type"])
            if key in existing_keys:
                stats["skipped_existing"] += 1
                continue
            batch.append(row)
            existing_keys.add(key)
            if len(batch) >= batch_size:
                if not dry_run:
                    supabase.table("atom_audit_evidence").insert(batch).execute()
                stats["inserted"] += len(batch)
                batch = []

    if batch:
        if not dry_run:
            supabase.table("atom_audit_evidence").insert(batch).execute()
        stats["inserted"] += len(batch)
    return stats


def build_uncertainty_rows(atom_id: str, estimates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map ``uncertainty.json`` estimates into insert rows."""
    rows: list[dict[str, Any]] = []
    for estimate in estimates:
        rows.append(
            {
                "atom_id": atom_id,
                "version_id": None,
                "mode": estimate.get("mode", "empirical"),
                "scalar_factor": estimate["scalar_factor"],
                "confidence": estimate["confidence"],
                "n_trials": estimate.get("n_trials", 0),
                "epsilon": estimate.get("epsilon", 0),
                "input_regime": estimate.get("input_regime", ""),
                "notes": estimate.get("notes", ""),
            }
        )
    return rows


def backfill_uncertainty(
    supabase: Any,
    *,
    atoms_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Populate ``atom_uncertainty_estimates`` from ``uncertainty.json`` files."""
    stats = {"found": 0, "inserted": 0, "skipped_no_atom": 0, "errors": 0}
    rows_by_atom_id: dict[str, list[dict[str, Any]]] = {}

    for uncertainty_path in iter_metadata_files("uncertainty.json", single_root=atoms_root):
        stats["found"] += 1
        try:
            payload = _load_json(uncertainty_path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read %s: %s", uncertainty_path, exc)
            stats["errors"] += 1
            continue

        atom_name = payload.get("atom", "")
        if not atom_name:
            stats["errors"] += 1
            continue
        namespace = namespace_from_path(uncertainty_path)
        atom_id = resolve_atom_id(supabase, namespace, atom_name)
        if not atom_id:
            stats["skipped_no_atom"] += 1
            continue
        try:
            rows = build_uncertainty_rows(atom_id, payload.get("estimates", []))
        except KeyError:
            stats["errors"] += 1
            continue

        rows_by_atom_id.setdefault(atom_id, []).extend(rows)

    for atom_id in sorted(rows_by_atom_id):
        batch = dedupe_uncertainty_rows(rows_by_atom_id[atom_id])
        if not dry_run:
            (
                supabase.table("atom_uncertainty_estimates")
                .delete()
                .eq("atom_id", atom_id)
                .is_("version_id", "null")
                .execute()
            )
            if batch:
                supabase.table("atom_uncertainty_estimates").insert(batch).execute()
        stats["inserted"] += len(batch)
    return stats


def normalize_verification_level(level: str) -> str:
    """Coerce unexpected verification levels to a schema-safe fallback."""
    return level if level in VALID_LEVELS else "unverified"


def build_verification_match_row(atom_id: str, match_result: dict[str, Any]) -> dict[str, Any]:
    """Map one ``matches.json`` entry into ``atom_verification_matches``."""
    pdg_node = match_result.get("pdg_node", {})
    verified_match = match_result.get("verified_match") or {}
    candidate = verified_match.get("candidate") or {}
    declaration = candidate.get("declaration") or {}
    return {
        "atom_id": atom_id,
        "version_id": None,
        "predicate_id": pdg_node.get("predicate_id", ""),
        "predicate_statement": pdg_node.get("statement", ""),
        "informal_desc": pdg_node.get("informal_desc", ""),
        "candidate_name": declaration.get("name", ""),
        "candidate_source_lib": declaration.get("source_lib", ""),
        "candidate_score": candidate.get("score"),
        "retrieval_method": candidate.get("retrieval_method", ""),
        "verified": verified_match.get("verified", False),
        "verification_level": normalize_verification_level(
            verified_match.get("verification_level", "unverified")
        ),
        "proof_term": verified_match.get("proof_term", ""),
        "compiler_output": verified_match.get("compiler_output", ""),
        "error_message": verified_match.get("error_message", ""),
        "all_candidates": match_result.get("all_candidates", []),
        "all_verifications": match_result.get("all_verifications", []),
    }


def backfill_verification_matches(
    supabase: Any,
    *,
    atoms_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Populate ``atom_verification_matches`` from ``matches.json`` files."""
    stats = {
        "files_found": 0,
        "entries_found": 0,
        "inserted": 0,
        "skipped_no_atom": 0,
        "errors": 0,
    }
    rows_by_atom_id: dict[str, list[dict[str, Any]]] = {}

    for matches_path in iter_metadata_files("matches.json", single_root=atoms_root):
        stats["files_found"] += 1
        try:
            entries = _load_json(matches_path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read %s: %s", matches_path, exc)
            stats["errors"] += 1
            continue
        if not isinstance(entries, list):
            stats["errors"] += 1
            continue

        namespace = namespace_from_path(matches_path)
        for entry in entries:
            stats["entries_found"] += 1
            predicate_id = (entry.get("pdg_node") or {}).get("predicate_id", "")
            if not predicate_id:
                stats["errors"] += 1
                continue
            atom_id = resolve_atom_id(supabase, namespace, predicate_id)
            if not atom_id:
                stats["skipped_no_atom"] += 1
                continue
            rows_by_atom_id.setdefault(atom_id, []).append(build_verification_match_row(atom_id, entry))

    for atom_id in sorted(rows_by_atom_id):
        batch = dedupe_verification_match_rows(rows_by_atom_id[atom_id])
        if not dry_run:
            (
                supabase.table("atom_verification_matches")
                .delete()
                .eq("atom_id", atom_id)
                .is_("version_id", "null")
                .execute()
            )
            if batch:
                supabase.table("atom_verification_matches").insert(batch).execute()
        stats["inserted"] += len(batch)
    return stats


def run_backfill_command(
    command: str,
    *,
    supabase: Any | None = None,
    dry_run: bool = False,
    atoms_root: Path | None = None,
    audit_manifest_path: Path | None = None,
    registry_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    runner_version: str = DEFAULT_RUNNER_VERSION,
) -> dict[str, Any]:
    """Dispatch one backfill command and return summary stats."""
    client = supabase or create_supabase_client_from_env()
    if command == "io-specs":
        return backfill_io_specs(
            client,
            atoms_root=atoms_root,
            audit_manifest_path=audit_manifest_path,
            dry_run=dry_run,
        )
    if command == "parameters":
        return backfill_parameters(
            client,
            audit_manifest_path=audit_manifest_path,
            dry_run=dry_run,
        )
    if command == "technical-descriptions":
        return backfill_technical_descriptions(
            client,
            audit_manifest_path=audit_manifest_path,
            dry_run=dry_run,
        )
    if command == "references-registry":
        return backfill_references_registry(
            client,
            registry_path=registry_path,
            dry_run=dry_run,
        )
    if command == "references":
        return backfill_references(
            client,
            atoms_root=atoms_root,
            registry_path=registry_path,
            dry_run=dry_run,
        )
    if command == "audit-rollups":
        return backfill_audit_rollups(
            client,
            manifest_path=audit_manifest_path,
            batch_size=batch_size,
            dry_run=dry_run,
        )
    if command == "audit-evidence":
        return backfill_audit_evidence(
            client,
            manifest_path=audit_manifest_path,
            batch_size=batch_size,
            runner_version=runner_version,
            dry_run=dry_run,
        )
    if command == "uncertainty":
        return backfill_uncertainty(
            client,
            atoms_root=atoms_root,
            dry_run=dry_run,
        )
    if command == "verification-matches":
        return backfill_verification_matches(
            client,
            atoms_root=atoms_root,
            dry_run=dry_run,
        )
    if command == "all-file-backed":
        summary: dict[str, Any] = {}
        ordered = (
            "references-registry",
            "io-specs",
            "parameters",
            "technical-descriptions",
            "references",
            "audit-rollups",
            "audit-evidence",
            "uncertainty",
            "verification-matches",
        )
        for item in ordered:
            summary[item] = run_backfill_command(
                item,
                supabase=client,
                dry_run=dry_run,
                atoms_root=atoms_root,
                audit_manifest_path=audit_manifest_path,
                registry_path=registry_path,
                batch_size=batch_size,
                runner_version=runner_version,
            )
        return summary
    raise ValueError(f"Unknown backfill command: {command}")
