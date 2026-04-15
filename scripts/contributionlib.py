#!/usr/bin/env python3
"""Shared helpers for atom contribution validation."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


VALID_HYPERPARAM_STATUSES = {"approved", "fixed", "blocked", "deprecated"}
VALID_HYPERPARAM_KINDS = {"int", "float", "categorical", "bool"}
VALID_SOURCE_CONFIDENCE = {"low", "medium", "high"}
JARGON_WORD_RE = re.compile(r"[A-Za-z]{2,}")
JARGON_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,}\b")


@dataclass(frozen=True)
class Finding:
    category: str
    severity: str
    path: str
    message: str
    line: int | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.line is None:
            payload.pop("line")
        return payload


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def add_finding(
    findings: list[Finding],
    *,
    category: str,
    severity: str,
    path: Path,
    repo_root: Path,
    message: str,
    line: int | None = None,
) -> None:
    findings.append(
        Finding(
            category=category,
            severity=severity,
            path=repo_relative(path, repo_root),
            message=message,
            line=line,
        )
    )


def parse_python(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def public_functions(module: ast.AST) -> list[ast.FunctionDef]:
    return [
        node
        for node in getattr(module, "body", [])
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]


def top_level_assignments(module: ast.AST) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in getattr(module, "body", []):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = evaluate_constant_expr(node.value, values)
        if value is not None:
            values[target.id] = value
    return values


def decorator_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = decorator_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    if isinstance(node, ast.Call):
        return decorator_name(node.func)
    return None


def direct_any_annotation(node: ast.AST | None) -> bool:
    if node is None:
        return False
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and inner.id == "Any":
            return True
        if isinstance(inner, ast.Attribute) and inner.attr == "Any":
            return True
    return False


def literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def evaluate_constant_expr(node: ast.AST, names: dict[str, object]) -> object | None:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.Tuple):
        items = [evaluate_constant_expr(item, names) for item in node.elts]
        if any(item is None for item in items):
            return None
        return tuple(items)
    if isinstance(node, ast.List):
        items = [evaluate_constant_expr(item, names) for item in node.elts]
        if any(item is None for item in items):
            return None
        return items
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
                continue
            if isinstance(value, ast.FormattedValue):
                inner = evaluate_constant_expr(value.value, names)
                if isinstance(inner, (str, int, float, bool)):
                    parts.append(str(inner))
                    continue
            return None
        return "".join(parts)
    return None


def module_path_to_file(repo_root: Path, module_name: str) -> Path | None:
    module_parts = module_name.split(".")
    module_file = repo_root.joinpath(*module_parts).with_suffix(".py")
    if module_file.exists():
        return module_file
    package_init = repo_root.joinpath(*module_parts, "__init__.py")
    if package_init.exists():
        return package_init
    return None


def function_names_in_module(path: Path) -> set[str]:
    module = parse_python(path)
    return {node.name for node in public_functions(module)}


def python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def atom_python_files(repo_root: Path) -> list[Path]:
    atoms_root = repo_root / "src" / "sciona" / "atoms"
    if not atoms_root.exists():
        return []
    return [
        path
        for path in python_files(atoms_root)
        if path.name != "__init__.py" and not path.name.startswith("_")
    ]


def witness_python_files(repo_root: Path) -> list[Path]:
    atoms_root = repo_root / "src" / "sciona" / "atoms"
    if not atoms_root.exists():
        return []
    return [
        path
        for path in python_files(atoms_root)
        if path.name == "witnesses.py" or path.name.endswith("_witnesses.py")
    ]


def probe_python_files(repo_root: Path) -> list[Path]:
    probes_root = repo_root / "src" / "sciona" / "probes"
    if not probes_root.exists():
        return []
    return [path for path in python_files(probes_root) if path.name != "__init__.py"]


def score_jargon(text: str) -> dict[str, float]:
    words = [word.lower() for word in JARGON_WORD_RE.findall(text)]
    if not words:
        return {
            "score": 0.0,
            "acronym_density": 0.0,
            "rare_word_ratio": 0.0,
            "flesch_kincaid_normalized": 0.0,
            "unexplained_acronym_ratio": 0.0,
        }

    common = common_words()
    acronym_density = len(JARGON_ACRONYM_RE.findall(text)) / len(words)
    rare_word_ratio = sum(1 for word in words if word not in common) / len(words)
    unexplained_acronym_ratio = compute_unexplained_acronym_ratio(text)
    fk = flesch_kincaid_grade(text)
    fk_normalized = min(fk / 18.0, 1.0)

    score = (
        0.20 * acronym_density
        + 0.35 * rare_word_ratio
        + 0.25 * fk_normalized
        + 0.20 * unexplained_acronym_ratio
    )
    return {
        "score": max(0.0, min(score, 1.0)),
        "acronym_density": acronym_density,
        "rare_word_ratio": rare_word_ratio,
        "flesch_kincaid_normalized": fk_normalized,
        "unexplained_acronym_ratio": unexplained_acronym_ratio,
    }


def compute_unexplained_acronym_ratio(text: str) -> float:
    acronyms = set(JARGON_ACRONYM_RE.findall(text))
    if not acronyms:
        return 0.0
    unexplained = 0
    for acronym in acronyms:
        pattern = re.compile(
            rf"\([^)]*{re.escape(acronym)}[^)]*\)"
            rf"|{re.escape(acronym)}\s*[-—–:]\s*\w",
            re.IGNORECASE,
        )
        if not pattern.search(text):
            unexplained += 1
    return unexplained / len(acronyms)


def flesch_kincaid_grade(text: str) -> float:
    sentences = [chunk.strip() for chunk in re.split(r"[.!?:]+", text) if chunk.strip()]
    words = JARGON_WORD_RE.findall(text)
    if not sentences or not words:
        return 0.0
    syllables = sum(count_syllables(word) for word in words)
    avg_words_per_sentence = len(words) / len(sentences)
    avg_syllables_per_word = syllables / len(words)
    return max(0.0, 0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word - 15.59)


def count_syllables(word: str) -> int:
    token = word.lower()
    if len(token) <= 2:
        return 1
    if token.endswith("e") and not token.endswith("le"):
        token = token[:-1]
    return max(1, len(re.findall(r"[aeiouy]+", token)))


_COMMON_WORDS: set[str] | None = None


def common_words() -> set[str]:
    global _COMMON_WORDS
    if _COMMON_WORDS is not None:
        return _COMMON_WORDS

    base = {
        "a", "about", "across", "add", "after", "all", "also", "an", "and", "any",
        "apply", "are", "array", "as", "at", "be", "because", "before", "both", "but",
        "by", "call", "can", "case", "change", "check", "class", "code", "common",
        "compute", "condition", "contains", "control", "convert", "could", "count",
        "create", "current", "data", "default", "define", "detect", "different", "do",
        "does", "each", "effect", "element", "else", "end", "ensure", "error", "even",
        "every", "example", "exist", "expected", "false", "field", "file", "filter",
        "first", "float", "for", "from", "function", "general", "generate", "get",
        "given", "good", "group", "has", "have", "help", "how", "if", "import", "in",
        "include", "index", "input", "int", "into", "is", "it", "its", "just", "keep",
        "key", "kind", "large", "last", "length", "less", "like", "line", "list",
        "load", "local", "long", "low", "make", "many", "map", "match", "may", "mean",
        "method", "minimum", "model", "module", "more", "most", "must", "name", "need",
        "new", "next", "no", "none", "not", "note", "number", "object", "of", "on",
        "one", "only", "open", "operation", "option", "or", "order", "other", "out",
        "output", "over", "pair", "parameter", "path", "pattern", "per", "perform",
        "point", "possible", "present", "process", "produce", "provides", "public",
        "raise", "range", "rate", "read", "real", "record", "reference", "result",
        "return", "returns", "rule", "run", "same", "sample", "scale", "score", "search",
        "set", "shape", "should", "show", "signal", "simple", "size", "small", "so",
        "some", "source", "specific", "state", "step", "still", "store", "string",
        "structure", "support", "summary", "system", "take", "target", "test", "text",
        "than", "that", "the", "their", "them", "then", "there", "these", "they",
        "this", "through", "time", "to", "true", "try", "tuple", "type", "under",
        "update", "use", "used", "uses", "using", "valid", "value", "values", "very",
        "way", "we", "well", "what", "when", "where", "which", "while", "why", "will",
        "with", "without", "work", "would", "write", "wrapper",
        "algorithm", "analysis", "atom", "atoms", "axis", "batch", "bound", "buffer",
        "cache", "callback", "coefficient", "config", "constraint", "coverage",
        "decorator", "deterministic", "distribution", "domain", "evidence", "family",
        "graph", "heuristic", "import", "interface", "matrix", "metadata", "monitor",
        "namespace", "numpy", "oracle", "probe", "registry", "runtime", "scalar",
        "semantic", "transition", "uncertainty", "vector", "witness",
    }

    dict_path = Path("/usr/share/dict/words")
    if dict_path.exists():
        try:
            for entry in dict_path.read_text(encoding="utf-8").splitlines():
                word = entry.strip().lower()
                if len(word) >= 2 and word.isalpha():
                    base.add(word)
        except OSError:
            pass

    _COMMON_WORDS = base
    return base


@dataclass(frozen=True)
class DocstringRecord:
    path: Path
    kind: str
    name: str
    text: str
    line: int


def collect_docstring_records(repo_root: Path) -> list[DocstringRecord]:
    records: list[DocstringRecord] = []
    for path in atom_python_files(repo_root):
        module = parse_python(path)
        module_docstring = ast.get_docstring(module, clean=False)
        if module_docstring:
            records.append(
                DocstringRecord(
                    path=path,
                    kind="module",
                    name=path.stem,
                    text=module_docstring,
                    line=1,
                )
            )
        for node in public_functions(module):
            docstring = ast.get_docstring(node, clean=False)
            if docstring:
                records.append(
                    DocstringRecord(
                        path=path,
                        kind="function",
                        name=node.name,
                        text=docstring,
                        line=node.lineno,
                    )
                )

    for path in repo_root.glob("data/heuristics/families/*.json"):
        try:
            payload = load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict):
            audit = payload.get("audit")
            if isinstance(audit, dict):
                dejargonized = audit.get("dejargonized_summary")
                if isinstance(dejargonized, str) and dejargonized.strip():
                    records.append(
                        DocstringRecord(
                            path=path,
                            kind="heuristic_audit",
                            name=payload.get("family", path.stem),
                            text=dejargonized,
                            line=1,
                        )
                    )

    for path in repo_root.rglob("heuristic_metadata.json"):
        try:
            payload = load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        records_list = payload.get("records")
        if not isinstance(records_list, list):
            continue
        for index, record in enumerate(records_list):
            if not isinstance(record, dict):
                continue
            dejargonized = record.get("dejargonized_summary")
            if isinstance(dejargonized, str) and dejargonized.strip():
                records.append(
                    DocstringRecord(
                        path=path,
                        kind="heuristic_record",
                        name=str(record.get("atom_fqdn", f"record[{index}]")),
                        text=dejargonized,
                        line=1,
                    )
                )
    return records


def summarize_findings(findings: Iterable[Finding]) -> dict[str, int]:
    summary = {"error": 0, "warning": 0}
    for finding in findings:
        summary[finding.severity] = summary.get(finding.severity, 0) + 1
    return summary


def json_dump(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
