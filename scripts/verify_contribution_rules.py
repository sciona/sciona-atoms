#!/usr/bin/env python3
"""Verify shared atom contribution rules for sibling sciona-atoms repos."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from contributionlib import (
    VALID_HYPERPARAM_KINDS,
    VALID_HYPERPARAM_STATUSES,
    VALID_SOURCE_CONFIDENCE,
    add_finding,
    atom_python_files,
    decorator_name,
    direct_any_annotation,
    function_names_in_module,
    json_dump,
    load_json,
    module_path_to_file,
    parse_python,
    probe_python_files,
    public_functions,
    summarize_findings,
    top_level_assignments,
    witness_python_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root to validate.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    parser.add_argument(
        "--fail-on",
        choices=("error", "warning", "none"),
        default="error",
        help="Severity threshold that should cause a non-zero exit.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()

    findings = []
    findings.extend(scan_interfaces(repo_root))
    findings.extend(scan_probes(repo_root))
    findings.extend(scan_hyperparams(repo_root))
    findings.extend(scan_references(repo_root))
    findings.extend(scan_heuristics(repo_root))

    summary = summarize_findings(findings)
    payload = {
        "repo_root": repo_root.as_posix(),
        "summary": summary,
        "findings": [finding.to_dict() for finding in findings],
    }

    if args.json:
        print(json_dump(payload))
    else:
        print(
            "contribution rules:"
            f" {summary.get('error', 0)} error(s), {summary.get('warning', 0)} warning(s)"
        )
        for finding in findings:
            location = finding.path if finding.line is None else f"{finding.path}:{finding.line}"
            print(f"- {finding.severity.upper()} [{finding.category}] {location}: {finding.message}")

    if args.fail_on == "warning" and findings:
        return 1
    if args.fail_on == "error" and any(finding.severity == "error" for finding in findings):
        return 1
    return 0


def scan_interfaces(repo_root: Path):
    findings = []
    witness_paths = set(witness_python_files(repo_root))

    for path in atom_python_files(repo_root):
        if path in witness_paths:
            continue

        module = parse_python(path)
        imported_witnesses = imported_witness_names(module)
        locally_defined = {node.name: node for node in public_functions(module)}
        for witness_name in imported_witnesses & locally_defined.keys():
            node = locally_defined[witness_name]
            add_finding(
                findings,
                category="interfaces",
                severity="error",
                path=path,
                repo_root=repo_root,
                line=node.lineno,
                message=f"local witness definition shadows imported witness `{witness_name}`",
            )

        contract_module = module_uses_contracts(module)
        register_atom_imported = module_imports_register_atom(module)

        for fn in public_functions(module):
            if fn.name.startswith("witness_"):
                continue
            validate_public_signature(
                findings,
                repo_root,
                path,
                fn,
                require_contracts=contract_module,
                require_register_atom=register_atom_imported,
            )
            if function_contains_non_ffi_notimplemented(fn):
                add_finding(
                    findings,
                    category="interfaces",
                    severity="error",
                    path=path,
                    repo_root=repo_root,
                    line=fn.lineno,
                    message="public atom function raises NotImplementedError",
                )

    for path in witness_python_files(repo_root):
        module = parse_python(path)
        for fn in public_functions(module):
            validate_witness_signature(findings, repo_root, path, fn)

    return findings


def module_uses_contracts(module: ast.AST) -> bool:
    for fn in public_functions(module):
        names = [decorator_name(node) for node in fn.decorator_list]
        if any(name and name.startswith("icontract.") for name in names):
            return True
        if "register_atom" in names:
            return True
    return False


def module_imports_register_atom(module: ast.AST) -> bool:
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.endswith("registry"):
                if any(alias.name == "register_atom" for alias in node.names):
                    return True
        if isinstance(node, ast.Import):
            if any(alias.name.endswith("register_atom") for alias in node.names):
                return True
    return False


def imported_witness_names(module: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module and "witness" in node.module:
            for alias in node.names:
                if alias.name.startswith("witness_"):
                    names.add(alias.asname or alias.name)
    return names


def validate_public_signature(
    findings: list,
    repo_root: Path,
    path: Path,
    fn: ast.FunctionDef,
    *,
    require_contracts: bool,
    require_register_atom: bool,
) -> None:
    if ast.get_docstring(fn) is None:
        add_finding(
            findings,
            category="interfaces",
            severity="error",
            path=path,
            repo_root=repo_root,
            line=fn.lineno,
            message=f"public function `{fn.name}` is missing a docstring",
        )

    if fn.returns is None:
        add_finding(
            findings,
            category="interfaces",
            severity="error",
            path=path,
            repo_root=repo_root,
            line=fn.lineno,
            message=f"public function `{fn.name}` is missing a return annotation",
        )

    if direct_any_annotation(fn.returns):
        add_finding(
            findings,
            category="interfaces",
            severity="warning",
            path=path,
            repo_root=repo_root,
            line=fn.lineno,
            message=f"public function `{fn.name}` exposes `Any` in its return annotation",
        )

    for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
        if arg.arg == "self":
            continue
        if arg.annotation is None:
            add_finding(
                findings,
                category="interfaces",
                severity="error",
                path=path,
                repo_root=repo_root,
                line=fn.lineno,
                message=f"parameter `{arg.arg}` on `{fn.name}` is missing a type annotation",
            )
        elif direct_any_annotation(arg.annotation):
            add_finding(
                findings,
                category="interfaces",
                severity="warning",
                path=path,
                repo_root=repo_root,
                line=fn.lineno,
                message=f"parameter `{arg.arg}` on `{fn.name}` exposes `Any` in its annotation",
            )

    if fn.args.vararg is not None:
        add_finding(
            findings,
            category="interfaces",
            severity="warning",
            path=path,
            repo_root=repo_root,
            line=fn.lineno,
            message=f"public function `{fn.name}` uses `*args`",
        )
    if fn.args.kwarg is not None:
        add_finding(
            findings,
            category="interfaces",
            severity="warning",
            path=path,
            repo_root=repo_root,
            line=fn.lineno,
            message=f"public function `{fn.name}` uses `**kwargs`",
        )

    decorator_names = [decorator_name(node) for node in fn.decorator_list]
    require_count = sum(name == "icontract.require" for name in decorator_names)
    ensure_count = sum(name == "icontract.ensure" for name in decorator_names)
    register_count = sum(name == "register_atom" for name in decorator_names)

    if require_contracts and require_count == 0:
        add_finding(
            findings,
            category="interfaces",
            severity="error",
            path=path,
            repo_root=repo_root,
            line=fn.lineno,
            message=f"contract-bearing public function `{fn.name}` is missing `@icontract.require`",
        )
    if require_contracts and ensure_count == 0:
        add_finding(
            findings,
            category="interfaces",
            severity="error",
            path=path,
            repo_root=repo_root,
            line=fn.lineno,
            message=f"contract-bearing public function `{fn.name}` is missing `@icontract.ensure`",
        )
    if require_register_atom and register_count == 0:
        add_finding(
            findings,
            category="interfaces",
            severity="warning",
            path=path,
            repo_root=repo_root,
            line=fn.lineno,
            message=f"public function `{fn.name}` is not decorated with `@register_atom(...)`",
        )
    if register_count and decorator_names[0] != "register_atom":
        add_finding(
            findings,
            category="interfaces",
            severity="error",
            path=path,
            repo_root=repo_root,
            line=fn.lineno,
            message=f"`@register_atom(...)` must be the outermost decorator on `{fn.name}`",
        )


def function_contains_non_ffi_notimplemented(fn: ast.FunctionDef) -> bool:
    if fn.name.startswith("_") and "ffi" in fn.name.lower():
        return False
    for node in ast.walk(fn):
        if not isinstance(node, ast.Raise):
            continue
        exc = node.exc
        if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "NotImplementedError":
            return True
        if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
            return True
    return False


def validate_witness_signature(findings: list, repo_root: Path, path: Path, fn: ast.FunctionDef) -> None:
    if not fn.name.startswith("witness_"):
        add_finding(
            findings,
            category="interfaces",
            severity="warning",
            path=path,
            repo_root=repo_root,
            line=fn.lineno,
            message=f"public witness helper `{fn.name}` does not use the `witness_` prefix",
        )

    if fn.returns is None:
        add_finding(
            findings,
            category="interfaces",
            severity="error",
            path=path,
            repo_root=repo_root,
            line=fn.lineno,
            message=f"witness `{fn.name}` is missing a return annotation",
        )
    for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
        if arg.annotation is None:
            add_finding(
                findings,
                category="interfaces",
                severity="error",
                path=path,
                repo_root=repo_root,
                line=fn.lineno,
                message=f"witness parameter `{arg.arg}` on `{fn.name}` is missing a type annotation",
            )
    if fn.args.vararg is not None:
        add_finding(
            findings,
            category="interfaces",
            severity="warning",
            path=path,
            repo_root=repo_root,
            line=fn.lineno,
            message=f"witness `{fn.name}` uses `*args`; replace it with an explicit signature before merge",
        )
    if fn.args.kwarg is not None:
        add_finding(
            findings,
            category="interfaces",
            severity="warning",
            path=path,
            repo_root=repo_root,
            line=fn.lineno,
            message=f"witness `{fn.name}` uses `**kwargs`; replace it with an explicit signature before merge",
        )
    if ast.get_docstring(fn) is None:
        add_finding(
            findings,
            category="interfaces",
            severity="warning",
            path=path,
            repo_root=repo_root,
            line=fn.lineno,
            message=f"witness `{fn.name}` is missing a docstring",
        )


def scan_probes(repo_root: Path):
    findings = []
    source_root = repo_root / "src"

    for path in probe_python_files(repo_root):
        module = parse_python(path)
        names = top_level_assignments(module)
        import_hits = 0
        probe_target_hits = 0

        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("sciona.atoms."):
                import_hits += 1
                target = module_path_to_file(source_root, node.module)
                if target is None:
                    add_finding(
                        findings,
                        category="probes",
                        severity="error",
                        path=path,
                        repo_root=repo_root,
                        line=node.lineno,
                        message=f"probe import target `{node.module}` does not exist",
                    )

            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ProbeTarget":
                probe_target_hits += 1
                record = extract_probe_target(node, names)
                if record is None:
                    add_finding(
                        findings,
                        category="probes",
                        severity="warning",
                        path=path,
                        repo_root=repo_root,
                        line=node.lineno,
                        message="probe target could not be evaluated statically",
                    )
                    continue
                module_import_path = record["module_import_path"]
                wrapper_symbol = record["wrapper_symbol"]
                atom_fqdn = record["atom_fqdn"]
                target = module_path_to_file(source_root, module_import_path)
                if target is None:
                    add_finding(
                        findings,
                        category="probes",
                        severity="error",
                        path=path,
                        repo_root=repo_root,
                        line=node.lineno,
                        message=f"probe target module `{module_import_path}` does not exist",
                    )
                    continue
                if not module_exports_symbol(target, wrapper_symbol):
                    add_finding(
                        findings,
                        category="probes",
                        severity="error",
                        path=path,
                        repo_root=repo_root,
                        line=node.lineno,
                        message=(
                            f"probe wrapper symbol `{wrapper_symbol}` does not exist in "
                            f"`{module_import_path}`"
                        ),
                    )
                if not atom_fqdn.endswith(f".{wrapper_symbol}"):
                    add_finding(
                        findings,
                        category="probes",
                        severity="warning",
                        path=path,
                        repo_root=repo_root,
                        line=node.lineno,
                        message="probe atom_fqdn does not end with the wrapper symbol",
                    )

        if import_hits == 0 and probe_target_hits == 0:
            add_finding(
                findings,
                category="probes",
                severity="warning",
                path=path,
                repo_root=repo_root,
                message="probe module does not declare atom imports or ProbeTarget records",
            )

    return findings


def extract_probe_target(node: ast.Call, names: dict[str, object]) -> dict[str, str] | None:
    if len(node.args) < 3:
        return None
    evaluated = [ast_literal(arg, names) for arg in node.args[:3]]
    if any(not isinstance(value, str) for value in evaluated):
        return None
    atom_fqdn, module_import_path, wrapper_symbol = evaluated
    return {
        "atom_fqdn": atom_fqdn,
        "module_import_path": module_import_path,
        "wrapper_symbol": wrapper_symbol,
    }


def ast_literal(node: ast.AST, names: dict[str, object]) -> object | None:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
                continue
            if isinstance(value, ast.FormattedValue):
                inner = ast_literal(value.value, names)
                if isinstance(inner, (str, int, float, bool)):
                    parts.append(str(inner))
                    continue
            return None
        return "".join(parts)
    return None


def module_exports_symbol(path: Path, symbol: str) -> bool:
    if symbol in function_names_in_module(path):
        return True

    if path.name != "__init__.py":
        return False

    module = parse_python(path)
    sibling_atoms = path.with_name("atoms.py")
    sibling_witnesses = path.with_name("witnesses.py")
    sibling_legacy = path.with_name(f"{path.parent.name}_witnesses.py")

    for node in getattr(module, "body", []):
        if not isinstance(node, ast.ImportFrom):
            continue
        imported_names = {alias.name for alias in node.names}
        if symbol in imported_names:
            return True
        if any(alias.name == "*" for alias in node.names):
            if node.level == 1 and node.module == "atoms" and sibling_atoms.exists() and symbol in function_names_in_module(sibling_atoms):
                return True
            if node.level == 1 and node.module == "witnesses" and sibling_witnesses.exists() and symbol in function_names_in_module(sibling_witnesses):
                return True
            if node.level == 1 and node.module and node.module.endswith("_witnesses") and sibling_legacy.exists() and symbol in function_names_in_module(sibling_legacy):
                return True

    if sibling_atoms.exists() and symbol in function_names_in_module(sibling_atoms):
        return True

    return False


def scan_hyperparams(repo_root: Path):
    findings = []
    manifest_path = repo_root / "data" / "hyperparams" / "manifest.json"
    if not manifest_path.exists():
        return findings

    try:
        payload = load_json(manifest_path)
    except json.JSONDecodeError as exc:
        add_finding(
            findings,
            category="hyperparameters",
            severity="error",
            path=manifest_path,
            repo_root=repo_root,
            message=f"invalid JSON: {exc}",
        )
        return findings

    if not isinstance(payload, dict):
        add_finding(
            findings,
            category="hyperparameters",
            severity="error",
            path=manifest_path,
            repo_root=repo_root,
            message="manifest must be a JSON object",
        )
        return findings

    reviewed = payload.get("reviewed_atoms")
    if not isinstance(reviewed, list):
        add_finding(
            findings,
            category="hyperparameters",
            severity="error",
            path=manifest_path,
            repo_root=repo_root,
            message="manifest must contain a `reviewed_atoms` list",
        )
        return findings

    seen_atoms = set()
    for index, record in enumerate(reviewed):
        line = index + 1
        if not isinstance(record, dict):
            add_finding(
                findings,
                category="hyperparameters",
                severity="error",
                path=manifest_path,
                repo_root=repo_root,
                line=line,
                message="each hyperparameter record must be an object",
            )
            continue

        atom_name = record.get("atom")
        if not isinstance(atom_name, str) or not atom_name.strip():
            add_finding(
                findings,
                category="hyperparameters",
                severity="error",
                path=manifest_path,
                repo_root=repo_root,
                line=line,
                message="each hyperparameter record needs a non-empty `atom`",
            )
            continue
        if atom_name in seen_atoms:
            add_finding(
                findings,
                category="hyperparameters",
                severity="error",
                path=manifest_path,
                repo_root=repo_root,
                line=line,
                message=f"duplicate hyperparameter record for `{atom_name}`",
            )
        seen_atoms.add(atom_name)

        status = record.get("status")
        if status not in VALID_HYPERPARAM_STATUSES:
            add_finding(
                findings,
                category="hyperparameters",
                severity="error",
                path=manifest_path,
                repo_root=repo_root,
                line=line,
                message=f"`{atom_name}` uses invalid status `{status}`",
            )

        path_value = record.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            add_finding(
                findings,
                category="hyperparameters",
                severity="error",
                path=manifest_path,
                repo_root=repo_root,
                line=line,
                message=f"`{atom_name}` is missing a source `path`",
            )
        else:
            target = repo_root / path_value
            if not target.exists():
                add_finding(
                    findings,
                    category="hyperparameters",
                    severity="error",
                    path=manifest_path,
                    repo_root=repo_root,
                    line=line,
                    message=f"`{atom_name}` points to missing path `{path_value}`",
                )

        tunables = record.get("tunable_params")
        blocked = record.get("blocked_params")
        if not isinstance(tunables, list) or not isinstance(blocked, list):
            add_finding(
                findings,
                category="hyperparameters",
                severity="error",
                path=manifest_path,
                repo_root=repo_root,
                line=line,
                message=f"`{atom_name}` must define `tunable_params` and `blocked_params` lists",
            )
            continue
        if status == "approved" and not tunables:
            add_finding(
                findings,
                category="hyperparameters",
                severity="error",
                path=manifest_path,
                repo_root=repo_root,
                line=line,
                message=f"`{atom_name}` is approved but has no tunable params",
            )

        for param in tunables:
            validate_tunable_param(findings, repo_root, manifest_path, atom_name, param, line)

    return findings


def validate_tunable_param(findings, repo_root: Path, manifest_path: Path, atom_name: str, param: object, line: int) -> None:
    if not isinstance(param, dict):
        add_finding(
            findings,
            category="hyperparameters",
            severity="error",
            path=manifest_path,
            repo_root=repo_root,
            line=line,
            message=f"`{atom_name}` has a non-object tunable parameter entry",
        )
        return

    name = param.get("name")
    kind = param.get("kind")
    safe_to_optimize = param.get("safe_to_optimize")
    if not isinstance(name, str) or not name.strip():
        add_finding(
            findings,
            category="hyperparameters",
            severity="error",
            path=manifest_path,
            repo_root=repo_root,
            line=line,
            message=f"`{atom_name}` has a tunable parameter without a valid `name`",
        )
    if kind not in VALID_HYPERPARAM_KINDS:
        add_finding(
            findings,
            category="hyperparameters",
            severity="error",
            path=manifest_path,
            repo_root=repo_root,
            line=line,
            message=f"`{atom_name}` parameter `{name}` uses invalid kind `{kind}`",
        )
    if not isinstance(safe_to_optimize, bool):
        add_finding(
            findings,
            category="hyperparameters",
            severity="error",
            path=manifest_path,
            repo_root=repo_root,
            line=line,
            message=f"`{atom_name}` parameter `{name}` must set boolean `safe_to_optimize`",
        )

    default = param.get("default")
    min_value = param.get("min_value")
    max_value = param.get("max_value")
    if isinstance(default, (int, float)) and isinstance(min_value, (int, float)) and isinstance(max_value, (int, float)):
        if min_value > max_value:
            add_finding(
                findings,
                category="hyperparameters",
                severity="error",
                path=manifest_path,
                repo_root=repo_root,
                line=line,
                message=f"`{atom_name}` parameter `{name}` has min_value > max_value",
            )
        if default < min_value or default > max_value:
            add_finding(
                findings,
                category="hyperparameters",
                severity="error",
                path=manifest_path,
                repo_root=repo_root,
                line=line,
                message=f"`{atom_name}` parameter `{name}` has default outside its bounds",
            )
    if "source_confidence" in param and param["source_confidence"] not in VALID_SOURCE_CONFIDENCE:
        add_finding(
            findings,
            category="hyperparameters",
            severity="error",
            path=manifest_path,
            repo_root=repo_root,
            line=line,
            message=f"`{atom_name}` parameter `{name}` uses invalid source_confidence",
        )


def scan_references(repo_root: Path):
    findings = []
    for path in sorted(repo_root.rglob("references.json")):
        if "__pycache__" in path.parts:
            continue
        try:
            payload = load_json(path)
        except json.JSONDecodeError as exc:
            add_finding(
                findings,
                category="references",
                severity="error",
                path=path,
                repo_root=repo_root,
                message=f"invalid JSON: {exc}",
            )
            continue
        if not isinstance(payload, dict):
            add_finding(
                findings,
                category="references",
                severity="error",
                path=path,
                repo_root=repo_root,
                message="references sidecar must be a JSON object",
            )
            continue
        atoms = payload.get("atoms")
        if not isinstance(atoms, dict) or not atoms:
            add_finding(
                findings,
                category="references",
                severity="error",
                path=path,
                repo_root=repo_root,
                message="references sidecar must contain a non-empty `atoms` mapping",
            )
            continue
        for atom_key, record in atoms.items():
            if not isinstance(atom_key, str) or not atom_key.strip():
                add_finding(
                    findings,
                    category="references",
                    severity="error",
                    path=path,
                    repo_root=repo_root,
                    message="references sidecar contains an invalid atom key",
                )
                continue
            if not isinstance(record, dict):
                add_finding(
                    findings,
                    category="references",
                    severity="error",
                    path=path,
                    repo_root=repo_root,
                    message=f"`{atom_key}` reference record must be an object",
                )
                continue
            # Validate that the atom key's file path and line number point to the
            # correct function definition. Keys have the format:
            #   fqdn@relative/path/to/atoms.py:line_number
            if "@" in atom_key and ":" in atom_key.split("@", 1)[1]:
                fqdn_part, location_part = atom_key.rsplit("@", 1)
                func_name = fqdn_part.rsplit(".", 1)[-1] if "." in fqdn_part else fqdn_part
                if ":" in location_part:
                    file_rel, line_str = location_part.rsplit(":", 1)
                    source_file = repo_root / "src" / file_rel
                    if source_file.exists() and line_str.isdigit():
                        line_no = int(line_str)
                        try:
                            lines = source_file.read_text(encoding="utf-8").splitlines()
                            if line_no < 1 or line_no > len(lines):
                                add_finding(
                                    findings,
                                    category="references",
                                    severity="error",
                                    path=path,
                                    repo_root=repo_root,
                                    message=f"`{atom_key}` references line {line_no} but file has {len(lines)} lines",
                                )
                            else:
                                target_line = lines[line_no - 1]
                                if f"def {func_name}" not in target_line:
                                    add_finding(
                                        findings,
                                        category="references",
                                        severity="error",
                                        path=path,
                                        repo_root=repo_root,
                                        message=(
                                            f"`{atom_key}` references line {line_no} but that line does not "
                                            f"contain `def {func_name}`. Line content: {target_line.strip()!r}"
                                        ),
                                    )
                        except OSError:
                            pass
                    elif not source_file.exists():
                        add_finding(
                            findings,
                            category="references",
                            severity="error",
                            path=path,
                            repo_root=repo_root,
                            message=f"`{atom_key}` references non-existent file `src/{file_rel}`",
                        )

            refs = record.get("references")
            if not isinstance(refs, list) or not refs:
                add_finding(
                    findings,
                    category="references",
                    severity="error",
                    path=path,
                    repo_root=repo_root,
                    message=f"`{atom_key}` must include at least one reference record",
                )
                continue
            for ref in refs:
                if not isinstance(ref, dict):
                    add_finding(
                        findings,
                        category="references",
                        severity="error",
                        path=path,
                        repo_root=repo_root,
                        message=f"`{atom_key}` contains a non-object reference entry",
                    )
                    continue
                if not isinstance(ref.get("ref_id"), str) or not ref["ref_id"].strip():
                    add_finding(
                        findings,
                        category="references",
                        severity="error",
                        path=path,
                        repo_root=repo_root,
                        message=f"`{atom_key}` has a reference without `ref_id`",
                    )
                metadata = ref.get("match_metadata")
                if metadata is not None:
                    if not isinstance(metadata, dict):
                        add_finding(
                            findings,
                            category="references",
                            severity="error",
                            path=path,
                            repo_root=repo_root,
                            message=f"`{atom_key}` has non-object `match_metadata`",
                        )
                    elif metadata.get("confidence") not in VALID_SOURCE_CONFIDENCE:
                        add_finding(
                            findings,
                            category="references",
                            severity="warning",
                            path=path,
                            repo_root=repo_root,
                            message=f"`{atom_key}` reference uses missing or invalid confidence metadata",
                        )
    return findings


def scan_heuristics(repo_root: Path):
    findings = []
    for path in sorted(repo_root.rglob("heuristic_metadata.json")):
        findings.extend(validate_heuristic_metadata(repo_root, path))

    family_root = repo_root / "data" / "heuristics"
    if family_root.exists():
        for path in sorted(family_root.rglob("*.json")):
            findings.extend(validate_family_heuristics(repo_root, path))
    return findings


def validate_heuristic_metadata(repo_root: Path, path: Path):
    findings = []
    try:
        payload = load_json(path)
    except json.JSONDecodeError as exc:
        add_finding(
            findings,
            category="heuristics",
            severity="error",
            path=path,
            repo_root=repo_root,
            message=f"invalid JSON: {exc}",
        )
        return findings

    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        add_finding(
            findings,
            category="heuristics",
            severity="error",
            path=path,
            repo_root=repo_root,
            message="heuristic metadata must contain a non-empty `records` list",
        )
        return findings

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            add_finding(
                findings,
                category="heuristics",
                severity="error",
                path=path,
                repo_root=repo_root,
                line=index + 1,
                message="heuristic metadata record must be an object",
            )
            continue
        for field in ("atom_fqdn", "summary", "dejargonized_summary"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                add_finding(
                    findings,
                    category="heuristics",
                    severity="error",
                    path=path,
                    repo_root=repo_root,
                    line=index + 1,
                    message=f"heuristic metadata record is missing `{field}`",
                )
        if not isinstance(record.get("heuristic_outputs"), list) or not record["heuristic_outputs"]:
            add_finding(
                findings,
                category="heuristics",
                severity="error",
                path=path,
                repo_root=repo_root,
                line=index + 1,
                message="heuristic metadata record needs a non-empty `heuristic_outputs` list",
            )
        else:
            for output in record["heuristic_outputs"]:
                if not isinstance(output, dict):
                    add_finding(
                        findings,
                        category="heuristics",
                        severity="error",
                        path=path,
                        repo_root=repo_root,
                        line=index + 1,
                        message="heuristic output entry must be an object",
                    )
                    continue
                heuristic = output.get("heuristic")
                if not isinstance(heuristic, dict):
                    add_finding(
                        findings,
                        category="heuristics",
                        severity="error",
                        path=path,
                        repo_root=repo_root,
                        line=index + 1,
                        message="heuristic output is missing nested `heuristic` metadata",
                    )
                    continue
                for field in ("heuristic_id", "display_name", "dejargonized_meaning"):
                    if not isinstance(heuristic.get(field), str) or not heuristic[field].strip():
                        add_finding(
                            findings,
                            category="heuristics",
                            severity="error",
                            path=path,
                            repo_root=repo_root,
                            line=index + 1,
                            message=f"heuristic output is missing `{field}`",
                        )
        if not isinstance(record.get("references"), list) or not record["references"]:
            add_finding(
                findings,
                category="heuristics",
                severity="error",
                path=path,
                repo_root=repo_root,
                line=index + 1,
                message="heuristic metadata record needs non-empty `references`",
            )
        if not isinstance(record.get("maintainers"), list) or not record["maintainers"]:
            add_finding(
                findings,
                category="heuristics",
                severity="error",
                path=path,
                repo_root=repo_root,
                line=index + 1,
                message="heuristic metadata record needs non-empty `maintainers`",
            )
    return findings


def validate_family_heuristics(repo_root: Path, path: Path):
    findings = []
    try:
        payload = load_json(path)
    except json.JSONDecodeError as exc:
        add_finding(
            findings,
            category="heuristics",
            severity="error",
            path=path,
            repo_root=repo_root,
            message=f"invalid JSON: {exc}",
        )
        return findings
    if not isinstance(payload, dict):
        add_finding(
            findings,
            category="heuristics",
            severity="error",
            path=path,
            repo_root=repo_root,
            message="family heuristic asset must be a JSON object",
        )
        return findings
    for field in ("asset_id", "family", "name", "summary"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            add_finding(
                findings,
                category="heuristics",
                severity="error",
                path=path,
                repo_root=repo_root,
                message=f"family heuristic asset is missing `{field}`",
            )
    bindings = payload.get("heuristic_bindings")
    if not isinstance(bindings, list) or not bindings:
        add_finding(
            findings,
            category="heuristics",
            severity="error",
            path=path,
            repo_root=repo_root,
            message="family heuristic asset needs a non-empty `heuristic_bindings` list",
        )
    else:
        for binding in bindings:
            if not isinstance(binding, dict):
                add_finding(
                    findings,
                    category="heuristics",
                    severity="error",
                    path=path,
                    repo_root=repo_root,
                    message="heuristic binding entry must be an object",
                )
                continue
            for field in ("heuristic_id",):
                if not isinstance(binding.get(field), str) or not binding[field].strip():
                    add_finding(
                        findings,
                        category="heuristics",
                        severity="error",
                        path=path,
                        repo_root=repo_root,
                        message=f"heuristic binding is missing `{field}`",
                    )
            for field in ("family_notes", "supported_action_classes", "uncertainty_notes", "references"):
                if not isinstance(binding.get(field), list) or not binding[field]:
                    add_finding(
                        findings,
                        category="heuristics",
                        severity="error",
                        path=path,
                        repo_root=repo_root,
                        message=f"heuristic binding is missing non-empty `{field}`",
                    )
    audit = payload.get("audit")
    if not isinstance(audit, dict):
        add_finding(
            findings,
            category="heuristics",
            severity="error",
            path=path,
            repo_root=repo_root,
            message="family heuristic asset needs an `audit` object",
        )
        return findings
    for field in ("dejargonized_summary",):
        if not isinstance(audit.get(field), str) or not audit[field].strip():
            add_finding(
                findings,
                category="heuristics",
                severity="error",
                path=path,
                repo_root=repo_root,
                message=f"family heuristic asset audit is missing `{field}`",
            )
    for field in ("uncertainty_notes", "references", "maintainers"):
        if not isinstance(audit.get(field), list) or not audit[field]:
            add_finding(
                findings,
                category="heuristics",
                severity="error",
                path=path,
                repo_root=repo_root,
                message=f"family heuristic asset audit is missing non-empty `{field}`",
            )
    return findings


if __name__ == "__main__":
    raise SystemExit(main())
