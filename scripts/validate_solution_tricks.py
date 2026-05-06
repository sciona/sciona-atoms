#!/usr/bin/env python3
"""Validate solution trick registry JSON files.

Usage:
    python scripts/validate_solution_tricks.py [--strict] [FILE ...]

Without arguments, validates data/solution_tricks/registry.json.
The validator intentionally uses only Python stdlib so provider repos can run it
without installing a JSON Schema package.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


VALID_KINDS = {
    "leak",
    "metric_hack",
    "postprocess",
    "data_artifact",
    "public_lb_overfit_risk",
    "domain_prior",
    "solver_shortcut",
    "inference_budget_trick",
}
VALID_STATUSES = {"cataloged", "allowed_with_validation", "deprecated", "disallowed"}
VALID_RISK_LEVELS = {"low", "medium", "high", "disallowed"}
VALID_GENERALIZATION_LEVELS = {"general", "domain_specific", "competition_specific"}
VALID_AUDIT_SOURCE_KINDS = {"manual_analysis", "kaggle_solution", "manual_hypothesis"}
VALID_AUDIT_REVIEW_STATUSES = {"draft", "reviewed", "deprecated"}

REQUIRED_TRICK_FIELDS = [
    "trick_id",
    "name",
    "kind",
    "status",
    "risk_level",
    "generalization_level",
    "summary",
    "applies_when",
    "do_not_use_when",
    "validation_requirements",
    "architect_hint",
    "related_cdgs",
    "related_operations",
    "source_competitions",
    "source_references",
    "tags",
    "audit",
]


class Violation:
    def __init__(self, file: str, level: str, message: str):
        self.file = file
        self.level = level
        self.message = message

    def __str__(self) -> str:
        tag = "ERROR" if self.level == "error" else "WARN "
        return f"  [{tag}] {self.file}: {self.message}"


def _load_json(path: Path) -> tuple[Any | None, list[Violation]]:
    violations: list[Violation] = []
    try:
        return json.loads(path.read_text()), violations
    except json.JSONDecodeError as exc:
        violations.append(Violation(path.name, "error", f"Invalid JSON: {exc}"))
        return None, violations


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _check_string_list(
    trick: dict[str, Any],
    field: str,
    violations: list[Violation],
    fname: str,
    *,
    allow_empty: bool = False,
) -> None:
    value = trick.get(field)
    tid = trick.get("trick_id", "<unknown>")
    if not isinstance(value, list):
        violations.append(Violation(fname, "error", f"{tid}.{field} must be a list"))
        return
    if not value and not allow_empty:
        violations.append(Violation(fname, "error", f"{tid}.{field} must not be empty"))
        return
    for i, item in enumerate(value):
        if not _is_non_empty_string(item):
            violations.append(Violation(fname, "error", f"{tid}.{field}[{i}] must be a non-empty string"))


def _solution_asset_ids(repo_root: Path) -> set[str]:
    asset_ids: set[str] = set()
    cdg_dir = repo_root / "data" / "solution_cdgs"
    if not cdg_dir.exists():
        return asset_ids
    for path in cdg_dir.glob("*.json"):
        if path.name.endswith("_bindings.json"):
            continue
        data, violations = _load_json(path)
        if violations or not isinstance(data, dict):
            continue
        asset_id = data.get("asset_id")
        if isinstance(asset_id, str):
            asset_ids.add(asset_id)
    return asset_ids


def validate_registry(path: Path, *, strict: bool = False, repo_root: Path | None = None) -> list[Violation]:
    violations: list[Violation] = []
    fname = path.name
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    solution_ids = _solution_asset_ids(repo_root)

    data, parse_violations = _load_json(path)
    violations.extend(parse_violations)
    if parse_violations:
        return violations

    if not isinstance(data, dict):
        return [Violation(fname, "error", "Root must be a JSON object")]

    if data.get("schema_version") != "v1":
        violations.append(Violation(fname, "error", "schema_version must be 'v1'"))

    tricks = data.get("tricks")
    if not isinstance(tricks, list):
        violations.append(Violation(fname, "error", "tricks must be a list"))
        return violations

    seen_ids: set[str] = set()
    seen_names: dict[str, str] = {}
    for i, trick in enumerate(tricks):
        if not isinstance(trick, dict):
            violations.append(Violation(fname, "error", f"tricks[{i}] must be an object"))
            continue

        tid = trick.get("trick_id", f"<tricks[{i}]>")
        for field in REQUIRED_TRICK_FIELDS:
            if field not in trick:
                violations.append(Violation(fname, "error", f"{tid} missing required field: {field}"))

        if not _is_non_empty_string(trick.get("trick_id")):
            violations.append(Violation(fname, "error", f"tricks[{i}].trick_id must be a non-empty string"))
            continue
        if not re.fullmatch(r"trick\.[a-z0-9_.-]+", trick["trick_id"]):
            violations.append(Violation(fname, "error", f"{tid}.trick_id must match trick.[a-z0-9_.-]+"))
        if trick["trick_id"] in seen_ids:
            violations.append(Violation(fname, "error", f"Duplicate trick_id: {trick['trick_id']}"))
        seen_ids.add(trick["trick_id"])
        if trick["trick_id"] in solution_ids:
            violations.append(Violation(fname, "error", f"{tid}.trick_id collides with a solution CDG asset_id"))

        for field in ["name", "summary", "architect_hint"]:
            if field in trick and not _is_non_empty_string(trick[field]):
                violations.append(Violation(fname, "error", f"{tid}.{field} must be a non-empty string"))
        if _is_non_empty_string(trick.get("name")):
            normalized_name = trick["name"].strip().lower()
            if normalized_name in seen_names:
                violations.append(
                    Violation(
                        fname,
                        "error",
                        f"Duplicate trick name: {trick['name']!r} also used by {seen_names[normalized_name]}",
                    )
                )
            seen_names[normalized_name] = trick.get("trick_id", tid)

        enum_checks = [
            ("kind", VALID_KINDS),
            ("status", VALID_STATUSES),
            ("risk_level", VALID_RISK_LEVELS),
            ("generalization_level", VALID_GENERALIZATION_LEVELS),
        ]
        for field, valid_values in enum_checks:
            if field in trick and trick[field] not in valid_values:
                violations.append(
                    Violation(fname, "error", f"{tid}.{field} has invalid value: {trick[field]!r}")
                )

        for field in [
            "applies_when",
            "do_not_use_when",
            "validation_requirements",
            "related_cdgs",
            "source_competitions",
            "source_references",
            "tags",
        ]:
            if field in trick:
                _check_string_list(trick, field, violations, fname)
        if "related_operations" in trick:
            _check_string_list(trick, "related_operations", violations, fname, allow_empty=True)

        if trick.get("risk_level") in {"high", "disallowed"} and not trick.get("validation_requirements"):
            violations.append(Violation(fname, "error", f"{tid} high/disallowed risk requires validation_requirements"))
        if trick.get("status") == "disallowed" and trick.get("risk_level") != "disallowed":
            violations.append(Violation(fname, "error", f"{tid} status disallowed requires risk_level disallowed"))

        for related_cdg in trick.get("related_cdgs", []):
            if isinstance(related_cdg, str) and related_cdg not in solution_ids:
                violations.append(Violation(fname, "error", f"{tid}.related_cdgs references unknown CDG: {related_cdg}"))

        audit = trick.get("audit")
        if not isinstance(audit, dict):
            violations.append(Violation(fname, "error", f"{tid}.audit must be an object"))
        else:
            for field in ["source_kind", "review_status", "notes"]:
                if field not in audit:
                    violations.append(Violation(fname, "error", f"{tid}.audit missing required field: {field}"))
            if audit.get("source_kind") not in VALID_AUDIT_SOURCE_KINDS:
                violations.append(Violation(fname, "error", f"{tid}.audit.source_kind has invalid value"))
            if audit.get("review_status") not in VALID_AUDIT_REVIEW_STATUSES:
                violations.append(Violation(fname, "error", f"{tid}.audit.review_status has invalid value"))
            if "notes" in audit and not _is_non_empty_string(audit["notes"]):
                violations.append(Violation(fname, "error", f"{tid}.audit.notes must be a non-empty string"))

        if strict and trick.get("audit", {}).get("source_kind") != "manual_hypothesis":
            refs = trick.get("source_references", [])
            if not refs:
                violations.append(Violation(fname, "error", f"{tid} requires source_references in strict mode"))

    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate solution trick registry JSON files")
    parser.add_argument("files", nargs="*", help="Registry JSON files to validate")
    parser.add_argument("--strict", action="store_true", help="Enable stricter source-reference checks")
    parser.add_argument("--errors-only", action="store_true", help="Suppress warnings, show only errors")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths = [Path(f) for f in args.files]
    if not paths:
        paths = [repo_root / "data" / "solution_tricks" / "registry.json"]

    total_errors = 0
    total_warnings = 0
    total_files = 0

    for path in paths:
        if not path.exists():
            print(f"  [ERROR] {path}: file not found")
            total_errors += 1
            continue

        violations = validate_registry(path, strict=args.strict, repo_root=repo_root)
        errors = [v for v in violations if v.level == "error"]
        warnings = [v for v in violations if v.level == "warning"]
        total_errors += len(errors)
        total_warnings += len(warnings)
        total_files += 1

        if errors or (warnings and not args.errors_only):
            for violation in violations:
                if args.errors_only and violation.level == "warning":
                    continue
                print(violation)

    print(f"\n{'=' * 60}")
    print(f"Validated {total_files} solution trick registry file(s)")
    print(f"  Errors:   {total_errors}")
    print(f"  Warnings: {total_warnings}")

    if total_errors:
        print(f"\nFAILED - {total_errors} errors must be fixed")
        sys.exit(1)
    if total_warnings:
        print(f"\nPASSED with {total_warnings} warnings")
        sys.exit(0)
    print("\nPASSED - all clean")


if __name__ == "__main__":
    main()
