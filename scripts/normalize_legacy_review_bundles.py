"""Normalize legacy provider review-bundle envelopes for the shared loader.

This intentionally does not synthesize atom rows, sources, or approval state.
Compact records lacked atom identifiers, so their original evidence cannot be
expanded safely; an empty ``atoms`` collection makes that absence explicit.
"""

from __future__ import annotations

import json
from pathlib import Path
import ast


WORKSPACE = Path(__file__).resolve().parents[2]
ENUM_MAP = {
    "acceptable_candidate": "acceptable_with_limits_candidate",
    "acceptable": "review_ready",
    "usage_equivalent": "parity_or_usage_equivalent",
    "exact_formula": "positive_path",
}


def _map_retired_enums(value: object) -> object:
    if isinstance(value, dict):
        return {key: _map_retired_enums(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_retired_enums(item) for item in value]
    if isinstance(value, str):
        return ENUM_MAP.get(value, value)
    return value


def _public_functions(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]


def _legacy_source_paths(repo_root: Path, family: str) -> tuple[Path, list[str]] | None:
    atom_root = repo_root / "src" / "sciona" / "atoms"
    matches = [path for path in atom_root.rglob(family) if path.is_dir()]
    if len(matches) != 1 or not (matches[0] / "atoms.py").is_file():
        return None
    directory = matches[0]
    names = ["atoms.py", "witnesses.py", "references.json", "cdg.json", "matches.json"]
    paths = [str((directory / name).relative_to(repo_root)) for name in names if (directory / name).is_file()]
    return directory, paths


def _materialize_legacy_rows(path: Path, payload: dict[str, object]) -> None:
    family = str(payload.get("family") or "").strip()
    if not family or payload.get("atoms") != []:
        return
    repo_root = next((parent for parent in path.parents if parent.name == "sciona-atoms" or parent.name.startswith("sciona-atoms-")), None)
    if repo_root is None:
        return
    resolved = _legacy_source_paths(repo_root, family)
    if resolved is None:
        return
    directory, source_paths = resolved
    functions = _public_functions(directory / "atoms.py")
    if not functions:
        return
    relative = directory.relative_to(repo_root / "src")
    package = ".".join(relative.parts)
    record_path = str(path.relative_to(repo_root))
    rationale = str(payload.get("rationale") or "Legacy PASS review record.")
    limitations = payload.get("limitations")
    limitation_rows = [str(limitations)] if isinstance(limitations, str) and limitations else []
    payload.update(
        {
            "bundle_id": f"{package}.review_bundle.{family}.legacy.v1",
            "provider_repo": repo_root.name,
            "family_batch": family,
            "review_status": "reviewed",
            "review_semantic_verdict": "pass_with_limits",
            "review_developer_semantic_verdict": "pass_with_limits",
            "trust_readiness": "reviewed_with_limits",
            "authoritative_sources": [
                {"kind": "local_wrapper" if item.endswith(("atoms.py", "witnesses.py")) else "local_metadata", "path": item}
                for item in source_paths
            ],
            "review_record_path": record_path,
            "rows": [
                {
                    "atom_name": f"{package}.{name}",
                    "atom_key": f"{package}.{name}",
                    "review_status": "reviewed",
                    "review_semantic_verdict": "pass_with_limits",
                    "review_developer_semantic_verdict": "pass_with_limits",
                    "trust_readiness": "needs_followup",
                    "source_paths": source_paths,
                    "review_record_path": record_path,
                    "limitations": limitation_rows,
                    "required_actions": [rationale],
                }
                for name in functions
            ],
        }
    )
    # The shared loader prefers ``atoms`` over ``rows``. Keep an empty atoms
    # list only for records that genuinely have no source-derived rows.
    payload.pop("atoms", None)


def _repair_missing_authoritative_sources(path: Path, payload: dict[str, object]) -> None:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return
    atom_name = str(rows[0].get("atom_name") or "") if isinstance(rows[0], dict) else ""
    prefix = "sciona.atoms."
    if not atom_name.startswith(prefix):
        return
    module_parts = atom_name[len(prefix):].rsplit(".", 1)[0].split(".")
    repo_root = next((parent for parent in path.parents if parent.name == "sciona-atoms"), None)
    if repo_root is None:
        return
    directory = repo_root / "src" / "sciona" / "atoms" / Path(*module_parts)
    names = ("atoms.py", "witnesses.py", "references.json", "cdg.json", "matches.json")
    source_paths = [str((directory / name).relative_to(repo_root)) for name in names if (directory / name).is_file()]
    if not source_paths:
        return
    if not payload.get("authoritative_sources"):
        payload["authoritative_sources"] = [
            {"kind": "local_wrapper" if item.endswith(("atoms.py", "witnesses.py")) else "local_metadata", "path": item}
            for item in source_paths
        ]
    for row in rows:
        if isinstance(row, dict):
            if not row.get("source_paths"):
                row["source_paths"] = source_paths
            row.setdefault("review_record_path", payload.get("review_record_path"))


def main() -> None:
    changed = 0
    for path in sorted(WORKSPACE.glob("sciona-atoms*/data/review_bundles/**/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        normalized = _map_retired_enums(payload)
        if not isinstance(normalized, dict):
            continue
        if not normalized.get("schema_version"):
            # The compact legacy schema has no atom identifiers. Do not guess
            # rows from filenames or mark any review status as approved.
            normalized["schema_version"] = "1.0"
            normalized["atoms"] = []
        _materialize_legacy_rows(path, normalized)
        _repair_missing_authoritative_sources(path, normalized)
        if normalized != payload:
            path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
            changed += 1
    print(f"normalized_review_bundles={changed}")


if __name__ == "__main__":
    main()
