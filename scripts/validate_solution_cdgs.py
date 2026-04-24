#!/usr/bin/env python3
"""Validate solution CDG JSON files against CDG_PROMPT.md quality rules.

Checks structural requirements, applicability completeness, stage/edge
consistency, DAG property, audit provenance, and reference attribution.

Usage:
    python validate_solution_cdgs.py [--strict] [FILE ...]

Without arguments, validates all CDGs in data/solution_cdgs/.
With --strict, applicability fields are required (not just recommended).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# ConceptType enum values (from sciona-matcher models.py)
# ---------------------------------------------------------------------------

VALID_CONCEPT_TYPES = {
    "sorting", "searching", "divide_and_conquer", "greedy",
    "dynamic_programming", "graph_traversal", "graph_optimization",
    "string_matching", "geometry", "arithmetic", "number_theory",
    "combinatorics", "algebra", "optimization", "analysis", "set_theory",
    "signal_transform", "signal_filter", "graph_signal_processing",
    "neural_network", "clustering", "dimensionality_reduction",
    "ode_solver", "quadrature", "randomized", "information_theory",
    "compression", "sampler", "log_prob", "posterior_update",
    "variational_inference", "prior_init", "prior_distribution",
    "likelihood_evaluation", "probabilistic_oracle", "oracle_gradient",
    "mcmc_kernel", "mcmc_proposal", "vi_elbo", "sequential_filter",
    "smc_reweight", "message_passing", "conjugate_update",
    "fixed_point", "map_over", "baseline_analysis", "ml_model_selection",
    "state_init", "data_assembly", "conditional_routing", "data_extraction",
    "visualization", "observability", "custom", "external_tool",
    "loss_function", "external_knowledge",
}

VALID_LOSS_CLASSES = {"preserving", "lossy_allowed", "lossy_but_allowed", "irreversible"}
VALID_EDGE_KINDS = {"data_flow", "callable_injection"}
VALID_SOURCE_KINDS = {"kaggle_solution", "drivendata_solution", "neurips_competition",
                      "kdd_cup", "recsys_challenge", "cikm_cup", "sigmod_contest",
                      "academic_benchmark", "manual_analysis"}


class Violation:
    def __init__(self, file: str, level: str, message: str):
        self.file = file
        self.level = level  # "error" or "warning"
        self.message = message

    def __str__(self):
        tag = "ERROR" if self.level == "error" else "WARN "
        return f"  [{tag}] {self.file}: {self.message}"


def validate_cdg(path: Path, strict: bool = False) -> list[Violation]:
    """Validate a single solution CDG JSON file."""
    violations: list[Violation] = []
    fname = path.name

    def error(msg: str):
        violations.append(Violation(fname, "error", msg))

    def warn(msg: str):
        violations.append(Violation(fname, "warning", msg))

    # --- JSON parse ---
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        error(f"Invalid JSON: {e}")
        return violations

    if not isinstance(data, dict):
        error("Root must be a JSON object")
        return violations

    # --- Top-level metadata ---
    for field in ["asset_id", "asset_version", "family", "paradigm", "name",
                  "summary", "dejargonized_summary"]:
        if field not in data:
            error(f"Missing required top-level field: {field}")
        elif not data[field]:
            error(f"Empty required field: {field}")

    if "asset_id" in data and not data["asset_id"].startswith("solution."):
        error(f"asset_id must start with 'solution.': got '{data['asset_id']}'")

    if "variant_hints" not in data or not isinstance(data.get("variant_hints"), list):
        error("Missing or non-list variant_hints")
    elif len(data.get("variant_hints", [])) < 2:
        warn("variant_hints should have at least 2 entries for effective retrieval")

    if "inputs" not in data or not isinstance(data.get("inputs"), list):
        error("Missing or non-list inputs")
    if "outputs" not in data or not isinstance(data.get("outputs"), list):
        error("Missing or non-list outputs")

    # --- Input/Output specs ---
    for io_list_name in ["inputs", "outputs"]:
        for i, io in enumerate(data.get(io_list_name, [])):
            if not isinstance(io, dict):
                error(f"{io_list_name}[{i}] is not an object")
                continue
            for req in ["name", "type_desc"]:
                if req not in io:
                    error(f"{io_list_name}[{i}] missing '{req}'")

    # --- Applicability block ---
    app = data.get("applicability")
    if app is None:
        if strict:
            error("Missing required 'applicability' block (strict mode)")
        else:
            warn("Missing 'applicability' block — add per CDG_PROMPT.md")
    elif isinstance(app, dict):
        for field in ["use_when", "do_not_use_when", "key_insight",
                      "critical_stages", "failure_modes"]:
            if field not in app:
                if strict:
                    error(f"applicability.{field} is required (strict mode)")
                else:
                    warn(f"applicability.{field} is recommended")

        if isinstance(app.get("use_when"), list) and len(app["use_when"]) < 2:
            warn("applicability.use_when should have at least 2 conditions")
        if isinstance(app.get("do_not_use_when"), list) and len(app["do_not_use_when"]) < 2:
            warn("applicability.do_not_use_when should have at least 2 conditions")

        key_insight = app.get("key_insight", "")
        if isinstance(key_insight, str) and len(key_insight) < 20:
            warn("applicability.key_insight is too short — explain WHY the pipeline works")

        # Validate critical_stages reference real stage IDs
        stage_ids = {s.get("stage_id") for s in data.get("stages", []) if isinstance(s, dict)}
        for cs in app.get("critical_stages", []):
            if cs not in stage_ids:
                error(f"applicability.critical_stages references unknown stage_id: '{cs}'")

        # Validate swappable_stages reference real stage IDs
        for ss in app.get("swappable_stages", {}).keys():
            if ss not in stage_ids:
                error(f"applicability.swappable_stages references unknown stage_id: '{ss}'")

    # --- Stages ---
    stages = data.get("stages", [])
    if not isinstance(stages, list):
        error("'stages' must be a list")
        stages = []

    if len(stages) < 2:
        warn("CDG has fewer than 2 stages — likely too simple for template matching")

    stage_ids = set()
    for i, stage in enumerate(stages):
        if not isinstance(stage, dict):
            error(f"stages[{i}] is not an object")
            continue

        sid = stage.get("stage_id", "")
        if not sid:
            error(f"stages[{i}] missing stage_id")
        elif sid in stage_ids:
            error(f"Duplicate stage_id: '{sid}'")
        stage_ids.add(sid)

        for req in ["name", "description", "concept_type"]:
            if req not in stage:
                error(f"Stage '{sid}' missing required field: {req}")

        ct = stage.get("concept_type", "")
        if ct and ct not in VALID_CONCEPT_TYPES:
            error(f"Stage '{sid}' has invalid concept_type: '{ct}'")

        if "dejargonized_description" not in stage:
            warn(f"Stage '{sid}' missing dejargonized_description")

        # Check inputs/outputs are lists of dicts with name+type_desc
        for io_field in ["inputs", "outputs"]:
            ios = stage.get(io_field, [])
            if not isinstance(ios, list):
                error(f"Stage '{sid}'.{io_field} must be a list")
            else:
                for j, io in enumerate(ios):
                    if isinstance(io, dict) and "name" not in io:
                        error(f"Stage '{sid}'.{io_field}[{j}] missing 'name'")

    # --- Edges ---
    edges = data.get("edges", [])
    if not isinstance(edges, list):
        error("'edges' must be a list")
        edges = []

    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            error(f"edges[{i}] is not an object")
            continue

        src = edge.get("source_stage_id", "")
        tgt = edge.get("target_stage_id", "")

        if src and src not in stage_ids:
            error(f"Edge [{i}] source_stage_id '{src}' not found in stages")
        if tgt and tgt not in stage_ids:
            error(f"Edge [{i}] target_stage_id '{tgt}' not found in stages")

        for req in ["source_stage_id", "target_stage_id"]:
            if req not in edge:
                error(f"edges[{i}] missing '{req}'")

        lc = edge.get("loss_class", "")
        if lc and lc not in VALID_LOSS_CLASSES:
            warn(f"Edge [{i}] unknown loss_class: '{lc}'")

        ek = edge.get("edge_kind", "")
        if ek and ek not in VALID_EDGE_KINDS:
            warn(f"Edge [{i}] unknown edge_kind: '{ek}'")

    # --- DAG check (cycle detection) ---
    adj: dict[str, list[str]] = {sid: [] for sid in stage_ids}
    for edge in edges:
        src = edge.get("source_stage_id", "")
        tgt = edge.get("target_stage_id", "")
        if src in adj and tgt in stage_ids:
            adj[src].append(tgt)

    visited: set[str] = set()
    in_stack: set[str] = set()
    has_cycle = False

    def dfs(node: str):
        nonlocal has_cycle
        if node in in_stack:
            has_cycle = True
            return
        if node in visited:
            return
        visited.add(node)
        in_stack.add(node)
        for neighbor in adj.get(node, []):
            dfs(neighbor)
        in_stack.discard(node)

    for sid in stage_ids:
        if sid not in visited:
            dfs(sid)
    if has_cycle:
        error("Stage graph contains a cycle — must be a DAG")

    # --- Planning constraints ---
    constraints = data.get("planning_constraints", [])
    if not isinstance(constraints, list):
        warn("'planning_constraints' should be a list")
    elif len(constraints) < 1:
        warn("No planning_constraints — add at least 1-2 ordering rules")
    else:
        for i, pc in enumerate(constraints):
            if not isinstance(pc, dict):
                continue
            for req in ["statement", "rationale"]:
                if req not in pc or not pc[req]:
                    warn(f"planning_constraints[{i}] missing or empty '{req}'")

    # --- Audit block ---
    audit = data.get("audit")
    if audit is None:
        error("Missing 'audit' block")
    elif isinstance(audit, dict):
        if "provenance_notes" not in audit:
            error("audit.provenance_notes is required — must include source URL")
        elif isinstance(audit["provenance_notes"], list):
            notes_text = " ".join(str(n) for n in audit["provenance_notes"])
            if not re.search(r"https?://", notes_text):
                warn("audit.provenance_notes should include a source URL (https://...)")

        if "references" not in audit or not audit.get("references"):
            warn("audit.references is empty — attribute the competition solution")

        if "maintainers" not in audit or not audit.get("maintainers"):
            warn("audit.maintainers is empty")

        sk = audit.get("source_kind", "")
        if sk and sk not in VALID_SOURCE_KINDS:
            warn(f"audit.source_kind '{sk}' not in known set — consider adding to validator")

    return violations


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate solution CDG JSON files")
    parser.add_argument(
        "files",
        nargs="*",
        help="CDG JSON files to validate (default: all in data/solution_cdgs/)",
    )
    parser.add_argument("--strict", action="store_true", help="Require applicability fields (not just recommend)")
    parser.add_argument("--errors-only", action="store_true", help="Suppress warnings, show only errors")
    args = parser.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        cdg_dir = Path(__file__).resolve().parents[1] / "data" / "solution_cdgs"
        paths = sorted(p for p in cdg_dir.glob("*.json") if "_bindings" not in p.name)

    total_errors = 0
    total_warnings = 0
    total_files = 0

    for path in paths:
        if not path.exists():
            print(f"  [ERROR] {path}: file not found")
            total_errors += 1
            continue

        violations = validate_cdg(path, strict=args.strict)
        errors = [v for v in violations if v.level == "error"]
        warnings = [v for v in violations if v.level == "warning"]
        total_errors += len(errors)
        total_warnings += len(warnings)
        total_files += 1

        if errors or (warnings and not args.errors_only):
            for v in violations:
                if args.errors_only and v.level == "warning":
                    continue
                print(v)

    # Summary
    print(f"\n{'='*60}")
    print(f"Validated {total_files} CDG files")
    print(f"  Errors:   {total_errors}")
    print(f"  Warnings: {total_warnings}")

    if total_errors > 0:
        print(f"\nFAILED — {total_errors} errors must be fixed")
        sys.exit(1)
    elif total_warnings > 0:
        print(f"\nPASSED with {total_warnings} warnings")
        sys.exit(0)
    else:
        print(f"\nPASSED — all clean")
        sys.exit(0)


if __name__ == "__main__":
    main()
