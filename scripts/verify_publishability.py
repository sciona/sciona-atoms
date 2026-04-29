#!/usr/bin/env python3
"""Verify that atoms in a family will be publishable without generating a manifest.

This script checks the five publishability pillars directly from source
artifacts (review bundles, CDG, references, atom source) without creating
or reading an audit_manifest.json.

Usage::

    # Check a specific family directory:
    python scripts/verify_publishability.py src/sciona/atoms/ml/g2p

    # Check from a sibling repo:
    cd /Users/conrad/personal/sciona-atoms-ml
    PYTHONPATH=src:/Users/conrad/personal/sciona-atoms/src \
      /Users/conrad/personal/sciona-matcher/.venv/bin/python \
      /Users/conrad/personal/sciona-atoms/scripts/verify_publishability.py \
      src/sciona/atoms/ml/g2p
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


def _find_atoms_in_module(family_dir: Path) -> list[str]:
    """Return list of atom FQDNs by inspecting atoms.py for @register_atom."""
    atoms_py = family_dir / "atoms.py"
    if not atoms_py.exists():
        return []
    fqdns: list[str] = []
    parts = list(family_dir.resolve().parts)
    try:
        src_idx = parts.index("src")
        module_parts = parts[src_idx + 1 :]
    except ValueError:
        return []
    module_prefix = ".".join(module_parts)
    source = atoms_py.read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") and "(" in stripped:
            func_name = stripped[4 : stripped.index("(")].strip()
            if not func_name.startswith("_"):
                fqdns.append(f"{module_prefix}.{func_name}")
    return fqdns


def _check_cdg(family_dir: Path, atom_fqdns: list[str]) -> list[str]:
    """Check CDG has concrete inputs/outputs for each atom."""
    errors: list[str] = []
    cdg_path = family_dir / "cdg.json"
    if not cdg_path.exists():
        errors.append(f"MISSING: {cdg_path.relative_to(family_dir.parent)}")
        return errors
    cdg = json.loads(cdg_path.read_text(encoding="utf-8"))
    nodes = {n["name"]: n for n in cdg.get("nodes", []) if "name" in n}
    for fqdn in atom_fqdns:
        func_name = fqdn.rsplit(".", 1)[-1]
        node = nodes.get(func_name)
        if node is None:
            errors.append(f"CDG: no node for {func_name}")
            continue
        if not node.get("inputs"):
            errors.append(f"CDG: {func_name} has no inputs")
        if not node.get("outputs"):
            errors.append(f"CDG: {func_name} has no outputs")
        for inp in node.get("inputs", []):
            if "name" not in inp or "type_desc" not in inp:
                errors.append(f"CDG: {func_name} input missing name/type_desc")
    return errors


def _check_references(family_dir: Path, atom_fqdns: list[str], repo_root: Path) -> list[str]:
    """Check references.json exists and all ref_ids resolve in local registry."""
    errors: list[str] = []
    refs_path = family_dir / "references.json"
    if not refs_path.exists():
        errors.append(f"MISSING: references.json")
        return errors
    refs = json.loads(refs_path.read_text(encoding="utf-8"))
    atoms_section = refs.get("atoms", {})
    leaf_names = {k.split("@")[0].rsplit(".", 1)[-1] for k in atoms_section}
    for fqdn in atom_fqdns:
        func_name = fqdn.rsplit(".", 1)[-1]
        if func_name not in leaf_names:
            errors.append(f"REFS: no entry for {func_name}")

    registry_path = repo_root / "data" / "references" / "registry.json"
    if not registry_path.exists():
        errors.append(f"MISSING: data/references/registry.json")
        return errors
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_ids = set(registry.get("references", {}).keys())
    for key, entry in atoms_section.items():
        for ref in entry.get("references", []):
            ref_id = ref.get("ref_id", "")
            if ref_id not in registry_ids:
                errors.append(f"REFS: ref_id '{ref_id}' not in local registry")
            meta = ref.get("match_metadata", {})
            if not meta.get("notes"):
                errors.append(f"REFS: {key} ref '{ref_id}' has empty notes")
    return errors


def _check_review_bundle(family_dir: Path, atom_fqdns: list[str], repo_root: Path) -> list[str]:
    """Check a review bundle exists and covers all atoms."""
    errors: list[str] = []
    bundle_dirs = [
        repo_root / "data" / "review_bundles",
        repo_root / "data" / "audit_reviews",
    ]
    bundles: list[dict[str, Any]] = []
    for bd in bundle_dirs:
        if not bd.is_dir():
            continue
        for path in sorted(bd.rglob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if "rows" in data:
                    bundles.append(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

    covered_atoms: set[str] = set()
    for bundle in bundles:
        for row in bundle.get("rows", []):
            name = row.get("atom_name") or row.get("atom_key") or ""
            covered_atoms.add(name)

    for fqdn in atom_fqdns:
        if fqdn not in covered_atoms:
            errors.append(f"BUNDLE: no review bundle row for {fqdn}")

    for bundle in bundles:
        for row in bundle.get("rows", []):
            name = row.get("atom_name", "")
            if name not in set(atom_fqdns):
                continue
            if not row.get("has_references"):
                errors.append(f"BUNDLE: {name} has_references is not true")
            if row.get("references_status") != "pass":
                errors.append(f"BUNDLE: {name} references_status is not 'pass'")
            verdict = row.get("review_semantic_verdict", "")
            if verdict not in ("pass", "pass_with_limits"):
                errors.append(f"BUNDLE: {name} review_semantic_verdict is '{verdict}'")
    return errors


def _check_atom_source(family_dir: Path, atom_fqdns: list[str]) -> list[str]:
    """Check atoms.py has register_atom, icontract.require, icontract.ensure."""
    errors: list[str] = []
    atoms_py = family_dir / "atoms.py"
    if not atoms_py.exists():
        errors.append("MISSING: atoms.py")
        return errors
    source = atoms_py.read_text(encoding="utf-8")
    if "register_atom" not in source:
        errors.append("SOURCE: no @register_atom decorator found")
    if "icontract.require" not in source and "@icontract.require" not in source:
        errors.append("SOURCE: no @icontract.require found")
    if "icontract.ensure" not in source and "@icontract.ensure" not in source:
        errors.append("SOURCE: no @icontract.ensure found")

    witnesses_py = family_dir / "witnesses.py"
    if not witnesses_py.exists():
        errors.append("MISSING: witnesses.py")

    init_py = family_dir / "__init__.py"
    if not init_py.exists():
        errors.append("MISSING: __init__.py")
    return errors


def _derive_repo_root(family_dir: Path) -> Path:
    """Walk up from family_dir to find the repo root (contains src/ or data/)."""
    current = family_dir.resolve()
    for parent in current.parents:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
        if parent.name == "sciona-atoms" or parent.name.startswith("sciona-atoms-"):
            return parent
    return family_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify atom publishability from source artifacts."
    )
    parser.add_argument(
        "family_dir",
        type=Path,
        help="Path to the atom family directory (e.g., src/sciona/atoms/ml/g2p)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (auto-detected if not given)",
    )
    args = parser.parse_args(argv)

    family_dir = args.family_dir.resolve()
    if not family_dir.is_dir():
        print(f"ERROR: {family_dir} is not a directory", file=sys.stderr)
        return 1

    repo_root = (args.repo_root or _derive_repo_root(family_dir)).resolve()

    atom_fqdns = _find_atoms_in_module(family_dir)
    if not atom_fqdns:
        print(f"ERROR: no public functions found in {family_dir / 'atoms.py'}", file=sys.stderr)
        return 1

    print(f"Repo root: {repo_root}")
    print(f"Family:    {family_dir.relative_to(repo_root)}")
    print(f"Atoms:     {len(atom_fqdns)}")
    for fqdn in atom_fqdns:
        print(f"  {fqdn}")
    print()

    all_errors: list[str] = []

    print("--- Source structure ---")
    errs = _check_atom_source(family_dir, atom_fqdns)
    all_errors.extend(errs)
    print(f"  {'PASS' if not errs else 'FAIL'}: {len(errs)} issue(s)")
    for e in errs:
        print(f"    {e}")

    print("--- CDG inputs/outputs ---")
    errs = _check_cdg(family_dir, atom_fqdns)
    all_errors.extend(errs)
    print(f"  {'PASS' if not errs else 'FAIL'}: {len(errs)} issue(s)")
    for e in errs:
        print(f"    {e}")

    print("--- References & registry ---")
    errs = _check_references(family_dir, atom_fqdns, repo_root)
    all_errors.extend(errs)
    print(f"  {'PASS' if not errs else 'FAIL'}: {len(errs)} issue(s)")
    for e in errs:
        print(f"    {e}")

    print("--- Review bundle coverage ---")
    errs = _check_review_bundle(family_dir, atom_fqdns, repo_root)
    all_errors.extend(errs)
    print(f"  {'PASS' if not errs else 'FAIL'}: {len(errs)} issue(s)")
    for e in errs:
        print(f"    {e}")

    print()
    if all_errors:
        print(f"RESULT: {len(all_errors)} publishability issue(s) found")
        return 1
    else:
        print("RESULT: all publishability checks passed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
