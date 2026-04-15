"""Provider-owned core Supabase seeding for sibling atom repositories.

This module intentionally covers only the base catalog spine needed before the
file-backed Phase 2 backfills run:

- discover sibling provider repositories
- derive repository provenance rows
- scan provider Python artifact roots for ``@register_atom`` definitions
- build ``atom_source_repositories`` and ``atoms`` upsert payloads
- optionally bootstrap a deterministic local owner through ``auth.users``

It also seeds provider-owned benchmark suites and flattens atom benchmark results
into the current relational tables while deferring CDG benchmark rows until the
unified artifact benchmark schema exists.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Iterable, Sequence
from uuid import NAMESPACE_URL, uuid5

from sciona.atoms.provider_inventory import (
    ProviderRepo,
    artifact_roots_for_repo,
    discover_provider_repos,
    namespace_prefix_for_artifact_root,
    workspace_root as provider_workspace_root,
)

logger = logging.getLogger(__name__)

DEFAULT_OWNER_LOGIN = "sciona-seed"
DEFAULT_OWNER_EMAIL = "sciona-seed@localhost"
DEFAULT_OWNER_DISPLAY_NAME = "Sciona Seed"
DEFAULT_OWNER_GITHUB_ID = -1
DEFAULT_STATUS = "approved"
DEFAULT_VISIBILITY_TIER = "general"
DEFAULT_SOURCE_KIND = "hand_written"
DEFAULT_STATEFUL_KIND = "none"
DEFERRED_TABLES: tuple[str, ...] = ()
_PY_FILE_STEM_OMIT = {"__init__", "atoms"}


@dataclass(frozen=True)
class ParsedAtomSpec:
    repo_name: str
    repo_root: Path
    artifact_root: Path
    file_path: Path
    registration_name: str
    witness_name: str
    module_name: str
    namespace_root: str
    namespace_path: str
    source_module_path: str
    source_symbol: str
    description: str
    domain_tags: tuple[str, ...]
    source_kind: str
    is_ffi: bool
    fqdn: str
    version_id: str
    content_hash: str
    semver: str
    s3_key: str
    fingerprint: str


@dataclass(frozen=True)
class RepositorySeedRow:
    repo_name: str
    repo_url: str
    namespace_root: str
    namespace_path: str = ""
    default_branch: str = "main"
    clone_priority: int = 100
    active: bool = True
    vcs_provider: str = "github"

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo_name": self.repo_name,
            "repo_url": self.repo_url,
            "namespace_root": self.namespace_root,
            "namespace_path": self.namespace_path,
            "default_branch": self.default_branch,
            "clone_priority": self.clone_priority,
            "active": self.active,
            "vcs_provider": self.vcs_provider,
        }


@dataclass(frozen=True)
class AtomSeedRow:
    fqdn: str
    namespace_root: str
    namespace_path: str
    repo_name: str
    source_package: str
    source_module_path: str
    source_symbol: str
    description: str
    domain_tags: tuple[str, ...]
    status: str = DEFAULT_STATUS
    visibility_tier: str = DEFAULT_VISIBILITY_TIER
    source_kind: str = DEFAULT_SOURCE_KIND
    stateful_kind: str = DEFAULT_STATEFUL_KIND
    is_stochastic: bool = False
    is_ffi: bool = False
    is_publishable: bool = False

    def as_dict(self, *, owner_id: str, source_repo_id: str | None = None) -> dict[str, Any]:
        return {
            "fqdn": self.fqdn,
            "namespace_root": self.namespace_root,
            "namespace_path": self.namespace_path,
            "owner_id": owner_id,
            "domain_tags": list(self.domain_tags),
            "description": self.description,
            "status": self.status,
            "visibility_tier": self.visibility_tier,
            "source_kind": self.source_kind,
            "stateful_kind": self.stateful_kind,
            "is_stochastic": self.is_stochastic,
            "is_ffi": self.is_ffi,
            "is_publishable": self.is_publishable,
            "source_repo_id": source_repo_id,
            "source_package": self.source_package,
            "source_module_path": self.source_module_path,
            "source_symbol": self.source_symbol,
        }


@dataclass(frozen=True)
class HyperparamSeedRow:
    fqdn: str
    name: str
    kind: str
    default_value: Any
    min_value: Any = None
    max_value: Any = None
    step_value: Any = None
    log_scale: bool = False
    choices_json: list[Any] | None = None
    constraints_json: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    semantic_role: str = ""
    status: str = "approved"

    def as_dict(self, *, atom_id: str) -> dict[str, Any]:
        return {
            "atom_id": atom_id,
            "name": self.name,
            "kind": self.kind,
            "default_value": self.default_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step_value": self.step_value,
            "log_scale": self.log_scale,
            "choices_json": self.choices_json,
            "constraints_json": self.constraints_json,
            "semantic_role": self.semantic_role,
            "status": self.status,
        }


@dataclass(frozen=True)
class VersionSeedRow:
    fqdn: str
    version_id: str
    content_hash: str
    semver: str
    is_latest: bool = True
    derives_from: str | None = None
    s3_key: str = ""
    fingerprint: str = ""

    def as_dict(self, *, atom_id: str) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "atom_id": atom_id,
            "content_hash": self.content_hash,
            "semver": self.semver,
            "is_latest": self.is_latest,
            "derives_from": self.derives_from,
            "s3_key": self.s3_key,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class BenchmarkSuiteSeedRow:
    benchmark_id: str
    domain_tags: tuple[str, ...]
    description: str
    dataset_s3_key: str = ""
    metric_names: tuple[str, ...] = ()
    curation_source: str = "foundation"
    status: str = "active"

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "domain_tags": list(self.domain_tags),
            "description": self.description,
            "dataset_s3_key": self.dataset_s3_key,
            "metric_names": list(self.metric_names),
            "curation_source": self.curation_source,
            "status": self.status,
        }


@dataclass(frozen=True)
class BenchmarkResultSeedRow:
    suite_id: str
    artifact_fqdn: str
    artifact_kind: str
    content_hash: str
    metric_name: str
    metric_value: float
    measured_at: str
    dataset_tag: str = ""
    semver: str = ""
    slice_key: str = ""
    runner: str = ""
    run_config_hash: str = ""
    status: str = "completed"
    evidence_uri: str = ""
    notes: str = ""

    def as_atom_benchmark_dict(self, *, version_id: str) -> dict[str, Any]:
        return {
            "version_id": version_id,
            "benchmark_name": self.suite_id,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "dataset_tag": self.dataset_tag,
            "measured_at": self.measured_at,
        }


@dataclass(frozen=True)
class OwnerSeed:
    user_id: str
    github_id: int
    github_login: str
    display_name: str
    avatar_url: str
    email: str

    def public_user_row(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "github_id": self.github_id,
            "github_login": self.github_login,
            "display_name": self.display_name,
            "avatar_url": self.avatar_url,
            "email": self.email,
            "identity_tier": "contributor",
            "stripe_account_id": None,
            "reputation_score": 0,
            "is_blacklisted": False,
            "effective_tier": "general",
        }


@dataclass(frozen=True)
class SeedInventory:
    base_dir: Path
    provider_repos: tuple[ProviderRepo, ...]
    repository_rows: tuple[RepositorySeedRow, ...]
    atom_rows: tuple[AtomSeedRow, ...]
    version_rows: tuple[VersionSeedRow, ...]
    hyperparam_rows: tuple[HyperparamSeedRow, ...]
    benchmark_suite_rows: tuple[BenchmarkSuiteSeedRow, ...]
    benchmark_result_rows: tuple[BenchmarkResultSeedRow, ...]
    parsed_atoms: int
    parsed_files: int
    duplicate_fqdns: tuple[str, ...] = ()
    empty_roots: tuple[str, ...] = ()
    deferred_tables: tuple[str, ...] = DEFERRED_TABLES

    def summary(self) -> dict[str, Any]:
        return {
            "provider_repos": len(self.provider_repos),
            "repository_rows": len(self.repository_rows),
            "atom_rows": len(self.atom_rows),
            "version_rows": len(self.version_rows),
            "hyperparam_rows": len(self.hyperparam_rows),
            "benchmark_suite_rows": len(self.benchmark_suite_rows),
            "benchmark_result_rows": len(self.benchmark_result_rows),
            "parsed_atoms": self.parsed_atoms,
            "parsed_files": self.parsed_files,
            "duplicate_fqdns": list(self.duplicate_fqdns),
            "empty_roots": list(self.empty_roots),
            "deferred_tables": list(self.deferred_tables),
        }


def _dedupe_preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _json_compatible_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, dict)):
        return value
    return str(value)


def _infer_hyperparam_kind(value: Any, choices: Sequence[Any] | None) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if choices:
        return "categorical"
    return "float"


def _git_remote_url(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return repo_root.resolve().as_uri()
    remote = (result.stdout or "").strip()
    return remote or repo_root.resolve().as_uri()


def _vcs_provider(repo_url: str) -> str:
    lowered = repo_url.lower()
    if "github" in lowered:
        return "github"
    if "gitlab" in lowered:
        return "gitlab"
    return "other"


def _iter_python_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _callable_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _register_atom_spec(decorator: ast.AST) -> tuple[str, str] | None:
    if isinstance(decorator, ast.Name) and decorator.id == "register_atom":
        return "", ""
    if not isinstance(decorator, ast.Call):
        return None
    if not isinstance(decorator.func, ast.Name) or decorator.func.id != "register_atom":
        return None
    witness_name = _callable_name(decorator.args[0]) if decorator.args else ""
    atom_name = ""
    for keyword in decorator.keywords:
        if keyword.arg != "name":
            continue
        try:
            value = ast.literal_eval(keyword.value)
        except Exception:
            value = None
        if isinstance(value, str):
            atom_name = value.strip()
            break
    return atom_name, witness_name


def _module_name_for_file(py_file: Path, artifact_root: Path) -> str:
    prefix = namespace_prefix_for_artifact_root(artifact_root)
    rel_parts = list(py_file.relative_to(artifact_root).with_suffix("").parts)
    if rel_parts and rel_parts[-1] in _PY_FILE_STEM_OMIT:
        rel_parts = rel_parts[:-1]
    return ".".join((*prefix, *rel_parts))


def _relative_module_path(module_name: str, namespace_root: str) -> str:
    if module_name == namespace_root:
        return ""
    return module_name.removeprefix(namespace_root).lstrip(".")


def _derive_fqdn(module_name: str, registration_name: str, namespace_root: str) -> str:
    if registration_name.startswith(namespace_root + "."):
        return registration_name
    if "." in registration_name:
        return ".".join(part for part in (namespace_root, registration_name) if part)
    if module_name:
        return f"{module_name}.{registration_name}"
    return ".".join(part for part in (namespace_root, registration_name) if part)


def _namespace_path_for_fqdn(fqdn: str, namespace_root: str) -> str:
    if fqdn == namespace_root:
        return ""
    suffix = fqdn.removeprefix(namespace_root).lstrip(".")
    parts = [part for part in suffix.split(".") if part]
    if len(parts) <= 1:
        return ""
    return ".".join(parts[:-1])


def _normalize_description(text: str) -> str:
    return " ".join(text.split()).strip()


def _infer_source_kind(path: Path) -> str:
    lowered = "/".join(path.parts).lower()
    if "_llm" in lowered or "/generated/" in lowered or "/ingest/" in lowered:
        return "generated_ingest"
    return DEFAULT_SOURCE_KIND


def _infer_domain_tags(namespace_path: str) -> tuple[str, ...]:
    first = namespace_path.split(".", 1)[0].strip()
    return (first,) if first else ()


class _AlphaRenamer(ast.NodeTransformer):
    """Normalize local binding names before hashing AST content."""

    def __init__(self) -> None:
        self._map: dict[str, str] = {}
        self._counter = 0

    def _canonical(self, name: str) -> str:
        if name not in self._map:
            self._map[name] = f"v{self._counter}"
            self._counter += 1
        return self._map[name]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = self._canonical(node.name)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._canonical(node.arg)
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._canonical(node.id)
        return node

    def visit_alias(self, node: ast.alias) -> ast.AST:
        if node.asname:
            node.asname = self._canonical(node.asname)
        return node


def _fingerprint_source(source: str) -> str:
    tree = ast.parse(source)
    tree = ast.fix_missing_locations(_AlphaRenamer().visit(tree))
    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalized_segment_source(source_text: str, node: ast.AST | None) -> str:
    if node is None:
        return ""
    segment = ast.get_source_segment(source_text, node) or ""
    return segment.strip()


def _version_material(
    *,
    repo_name: str,
    file_path: Path,
    fqdn: str,
    source_symbol: str,
    witness_name: str,
    source_text: str,
    node: ast.AST,
    witness: ast.AST | None,
) -> tuple[str, str, str, str, str]:
    function_source = _normalized_segment_source(source_text, node)
    witness_source = _normalized_segment_source(source_text, witness)
    combined_source = "\n\n".join(part for part in (witness_source, function_source) if part).strip()
    if not combined_source:
        combined_source = source_text
    fingerprint = _fingerprint_source(combined_source)
    payload = json.dumps(
        {
            "fqdn": fqdn,
            "repo_name": repo_name,
            "relative_path": str(file_path),
            "source_symbol": source_symbol,
            "witness_name": witness_name,
            "fingerprint": fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    version_id = str(uuid5(NAMESPACE_URL, f"sciona-local-version:{fqdn}:{content_hash}"))
    semver = f"0.0.0+local.{content_hash[:12]}"
    s3_key = f"local-seed/atoms/{content_hash}.tar.gz"
    return version_id, content_hash, semver, s3_key, fingerprint


def _parse_registered_atoms(
    *,
    repo: ProviderRepo,
    artifact_root: Path,
) -> tuple[ParsedAtomSpec, ...]:
    namespace_root = ".".join(namespace_prefix_for_artifact_root(artifact_root))
    parsed: list[ParsedAtomSpec] = []
    for py_file in _iter_python_files(artifact_root):
        try:
            source_text = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source_text, filename=str(py_file))
        except Exception:
            logger.debug("Failed to parse %s", py_file, exc_info=True)
            continue
        module_name = _module_name_for_file(py_file, artifact_root)
        function_defs = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            spec: tuple[str, str] | None = None
            for decorator in node.decorator_list:
                spec = _register_atom_spec(decorator)
                if spec is not None:
                    break
            if spec is None:
                continue
            registration_name, witness_name = spec
            registration_name = registration_name or node.name
            witness = function_defs.get(witness_name)
            description = _normalize_description(
                (ast.get_docstring(node) or "")
                or (ast.get_docstring(witness) or "") if witness is not None else ""
            )
            fqdn = _derive_fqdn(module_name, registration_name, namespace_root)
            namespace_path = _namespace_path_for_fqdn(fqdn, namespace_root)
            version_id, content_hash, semver, s3_key, fingerprint = _version_material(
                repo_name=repo.repo_name,
                file_path=py_file.relative_to(repo.repo_root),
                fqdn=fqdn,
                source_symbol=node.name,
                witness_name=witness_name,
                source_text=source_text,
                node=node,
                witness=witness,
            )
            parsed.append(
                ParsedAtomSpec(
                    repo_name=repo.repo_name,
                    repo_root=repo.repo_root,
                    artifact_root=artifact_root,
                    file_path=py_file,
                    registration_name=registration_name,
                    witness_name=witness_name,
                    module_name=module_name,
                    namespace_root=namespace_root,
                    namespace_path=namespace_path,
                    source_module_path=_relative_module_path(module_name, namespace_root),
                    source_symbol=node.name,
                    description=description,
                    domain_tags=_infer_domain_tags(namespace_path),
                    source_kind=_infer_source_kind(py_file),
                    is_ffi=node.name.endswith("_ffi"),
                    fqdn=fqdn,
                    version_id=version_id,
                    content_hash=content_hash,
                    semver=semver,
                    s3_key=s3_key,
                    fingerprint=fingerprint,
                )
            )
    return tuple(parsed)


def _discover_repo_artifact_roots(
    base_dir: Path | None = None,
) -> list[tuple[int, ProviderRepo, Path]]:
    provider_repos = discover_provider_repos(base_dir)
    roots: list[tuple[int, ProviderRepo, Path]] = []
    for priority, repo in enumerate(provider_repos):
        for artifact_root in artifact_roots_for_repo(repo.repo_root):
            roots.append((priority, repo, artifact_root))
    roots.sort(key=lambda item: (item[0], item[1].repo_name, str(item[2])))
    return roots


def _artifact_root_for_source_path(repo_root: Path, relative_path: str) -> Path | None:
    source_path = (repo_root / relative_path).resolve()
    for artifact_root in artifact_roots_for_repo(repo_root):
        try:
            source_path.relative_to(artifact_root)
            return artifact_root
        except ValueError:
            continue
    return None


def _derive_hyperparam_fqdn(repo_root: Path, entry: dict[str, Any]) -> str | None:
    atom_name = str(entry.get("atom") or "").strip()
    if not atom_name:
        return None
    if atom_name.startswith("ageoa.") or atom_name.startswith("sciona.atoms."):
        return atom_name
    relative_path = str(entry.get("path") or "").strip()
    if not relative_path:
        return atom_name
    artifact_root = _artifact_root_for_source_path(repo_root, relative_path)
    if artifact_root is None:
        return atom_name
    module_name = _module_name_for_file((repo_root / relative_path).resolve(), artifact_root)
    namespace_root = ".".join(namespace_prefix_for_artifact_root(artifact_root))
    return _derive_fqdn(module_name, atom_name, namespace_root)


def _hyperparam_rank(row: HyperparamSeedRow) -> tuple[int, int, str]:
    return (
        1 if row.semantic_role else 0,
        len(row.choices_json or []),
        json.dumps(row.as_dict(atom_id="placeholder"), sort_keys=True, separators=(",", ":")),
    )


def _derive_hyperparam_rows(base_dir: Path | None = None) -> tuple[HyperparamSeedRow, ...]:
    rows_by_key: dict[tuple[str, str], HyperparamSeedRow] = {}
    for repo in discover_provider_repos(base_dir):
        manifest_path = repo.repo_root / "data" / "hyperparams" / "manifest.json"
        if not manifest_path.is_file():
            continue
        payload = json.loads(manifest_path.read_text())
        atoms = payload.get("reviewed_atoms", []) if isinstance(payload, dict) else []
        if not isinstance(atoms, list):
            continue
        for atom in atoms:
            if not isinstance(atom, dict):
                continue
            if str(atom.get("status") or "") != "approved":
                continue
            fqdn = _derive_hyperparam_fqdn(repo.repo_root, atom)
            if not fqdn:
                continue
            tunables = atom.get("tunable_params", [])
            if not isinstance(tunables, list):
                continue
            for raw_param in tunables:
                if not isinstance(raw_param, dict):
                    continue
                if raw_param.get("safe_to_optimize", True) is False:
                    continue
                name = str(raw_param.get("name") or "").strip()
                if not name:
                    continue
                choices = raw_param.get("choices")
                choices_json = list(choices) if isinstance(choices, list) and choices else None
                constraints = raw_param.get("constraints")
                if constraints is not None and not isinstance(
                    constraints, (dict, list, str, int, float, bool)
                ):
                    constraints = str(constraints)
                row = HyperparamSeedRow(
                    fqdn=fqdn,
                    name=name,
                    kind=str(
                        raw_param.get("kind")
                        or _infer_hyperparam_kind(raw_param.get("default"), choices_json)
                    ),
                    default_value=_json_compatible_value(raw_param.get("default")),
                    min_value=_json_compatible_value(raw_param.get("min_value", raw_param.get("min"))),
                    max_value=_json_compatible_value(raw_param.get("max_value", raw_param.get("max"))),
                    step_value=_json_compatible_value(raw_param.get("step_value", raw_param.get("step"))),
                    log_scale=bool(raw_param.get("log_scale", False)),
                    choices_json=[_json_compatible_value(value) for value in choices_json] if choices_json else None,
                    constraints_json=_json_compatible_value(constraints),
                    semantic_role=str(raw_param.get("semantic_role") or "").strip(),
                )
                key = (row.fqdn, row.name)
                incumbent = rows_by_key.get(key)
                if incumbent is None or _hyperparam_rank(row) > _hyperparam_rank(incumbent):
                    rows_by_key[key] = row
    return tuple(rows_by_key[key] for key in sorted(rows_by_key))


_BENCHMARK_SUITE_STATUS = {"draft": "proposed", "active": "active", "retired": "retired"}
_ALLOWED_ARTIFACT_SCOPES = {"atom", "cdg", "both"}
_ALLOWED_RESULT_STATUSES = {"completed", "failed", "partial"}
_ALLOWED_ARTIFACT_KINDS = {"atom", "cdg"}


def _benchmark_description(title: str, contract_summary: str) -> str:
    title = _normalize_description(title)
    contract_summary = _normalize_description(contract_summary)
    if title and contract_summary:
        return f"{title}. {contract_summary}"
    return title or contract_summary


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in {path}")
    return [entry for entry in payload if isinstance(entry, dict)]


def _derive_benchmark_rows(base_dir: Path | None = None) -> tuple[tuple[BenchmarkSuiteSeedRow, ...], tuple[BenchmarkResultSeedRow, ...]]:
    suite_rows: dict[str, BenchmarkSuiteSeedRow] = {}
    suite_metric_names: dict[str, tuple[str, ...]] = {}
    suite_dataset_tags: dict[str, str] = {}
    result_rows: list[BenchmarkResultSeedRow] = []

    for repo in discover_provider_repos(base_dir):
        benchmark_dir = repo.repo_root / "data" / "benchmarks"
        suites_path = benchmark_dir / "benchmark_suites.json"
        results_path = benchmark_dir / "benchmark_results.json"
        if not suites_path.is_file() and not results_path.is_file():
            continue

        if suites_path.is_file():
            for raw_suite in _load_json_list(suites_path):
                suite_id = str(raw_suite.get("suite_id") or "").strip()
                if not suite_id:
                    raise ValueError(f"Benchmark suite missing suite_id in {suites_path}")
                if suite_id in suite_rows:
                    raise ValueError(f"Duplicate benchmark suite_id {suite_id} in {suites_path}")
                artifact_scope = str(raw_suite.get("artifact_scope") or "both").strip()
                if artifact_scope not in _ALLOWED_ARTIFACT_SCOPES:
                    raise ValueError(f"Invalid artifact_scope {artifact_scope!r} for {suite_id}")
                raw_metrics = raw_suite.get("metrics")
                if not isinstance(raw_metrics, list) or not raw_metrics:
                    raise ValueError(f"Benchmark suite {suite_id} must declare at least one metric")
                metric_names: list[str] = []
                primary_metrics = 0
                for raw_metric in raw_metrics:
                    if not isinstance(raw_metric, dict):
                        raise ValueError(f"Benchmark suite {suite_id} has non-object metric entry")
                    metric_name = str(raw_metric.get("metric_name") or "").strip()
                    if not metric_name:
                        raise ValueError(f"Benchmark suite {suite_id} has metric with empty metric_name")
                    if metric_name in metric_names:
                        raise ValueError(f"Benchmark suite {suite_id} repeats metric {metric_name}")
                    metric_names.append(metric_name)
                    if bool(raw_metric.get("primary", False)):
                        primary_metrics += 1
                if primary_metrics > 1:
                    raise ValueError(f"Benchmark suite {suite_id} declares more than one primary metric")
                suite_status = str(raw_suite.get("status") or "active").strip().lower()
                if suite_status not in _BENCHMARK_SUITE_STATUS:
                    raise ValueError(f"Invalid benchmark suite status {suite_status!r} for {suite_id}")
                domain_tags = _dedupe_preserve_order(
                    [
                        *list(raw_suite.get("domain_tags") or []),
                        *list(raw_suite.get("family_tags") or []),
                        *list(raw_suite.get("modality_tags") or []),
                    ]
                )
                suite_rows[suite_id] = BenchmarkSuiteSeedRow(
                    benchmark_id=suite_id,
                    domain_tags=domain_tags,
                    description=_benchmark_description(
                        str(raw_suite.get("title") or ""),
                        str(raw_suite.get("contract_summary") or ""),
                    ),
                    dataset_s3_key=str(raw_suite.get("dataset_s3_key") or "").strip(),
                    metric_names=tuple(metric_names),
                    curation_source="foundation",
                    status=_BENCHMARK_SUITE_STATUS[suite_status],
                )
                suite_metric_names[suite_id] = tuple(metric_names)
                suite_dataset_tags[suite_id] = str(raw_suite.get("dataset_tag") or "").strip()

        if results_path.is_file():
            for raw_result in _load_json_list(results_path):
                suite_id = str(raw_result.get("suite_id") or "").strip()
                if suite_id not in suite_metric_names:
                    raise ValueError(f"Benchmark result references unknown suite_id {suite_id!r} in {results_path}")
                metric_name = str(raw_result.get("metric_name") or "").strip()
                if metric_name not in suite_metric_names[suite_id]:
                    raise ValueError(
                        f"Benchmark result for {suite_id} references undeclared metric {metric_name!r} in {results_path}"
                    )
                artifact_kind = str(raw_result.get("artifact_kind") or "").strip()
                if artifact_kind not in _ALLOWED_ARTIFACT_KINDS:
                    raise ValueError(f"Invalid artifact_kind {artifact_kind!r} in {results_path}")
                status = str(raw_result.get("status") or "completed").strip()
                if status not in _ALLOWED_RESULT_STATUSES:
                    raise ValueError(f"Invalid benchmark result status {status!r} in {results_path}")
                artifact_fqdn = str(raw_result.get("artifact_fqdn") or "").strip()
                content_hash = str(raw_result.get("content_hash") or "").strip()
                if not artifact_fqdn or not content_hash:
                    raise ValueError(f"Benchmark result in {results_path} must include artifact_fqdn and content_hash")
                try:
                    metric_value = float(raw_result.get("metric_value"))
                except Exception as exc:
                    raise ValueError(f"Benchmark result for {suite_id} has invalid metric_value") from exc
                result_rows.append(
                    BenchmarkResultSeedRow(
                        suite_id=suite_id,
                        artifact_fqdn=artifact_fqdn,
                        artifact_kind=artifact_kind,
                        content_hash=content_hash,
                        semver=str(raw_result.get("semver") or "").strip(),
                        metric_name=metric_name,
                        metric_value=metric_value,
                        slice_key=str(raw_result.get("slice_key") or "").strip(),
                        measured_at=str(raw_result.get("measured_at") or "").strip(),
                        runner=str(raw_result.get("runner") or "").strip(),
                        run_config_hash=str(raw_result.get("run_config_hash") or "").strip(),
                        status=status,
                        evidence_uri=str(raw_result.get("evidence_uri") or "").strip(),
                        notes=str(raw_result.get("notes") or "").strip(),
                        dataset_tag=suite_dataset_tags.get(suite_id, ""),
                    )
                )

    ordered_suites = tuple(suite_rows[key] for key in sorted(suite_rows))
    ordered_results = tuple(
        sorted(
            result_rows,
            key=lambda row: (row.suite_id, row.artifact_kind, row.artifact_fqdn, row.metric_name, row.slice_key, row.content_hash),
        )
    )
    return ordered_suites, ordered_results


def _looks_like_workspace_root(path: Path) -> bool:
    try:
        entries = {child.name for child in path.iterdir() if child.is_dir()}
    except Exception:
        return False
    return "ageo-atoms" in entries or any(name.startswith("sciona-atoms") for name in entries)


def _resolve_workspace_root(base_dir: Path | None = None) -> Path:
    if base_dir is not None:
        return base_dir.expanduser().resolve()
    cwd = Path.cwd().expanduser().resolve()
    for candidate in (cwd, cwd.parent, provider_workspace_root()):
        if _looks_like_workspace_root(candidate):
            return candidate.resolve()
    return cwd


def derive_seed_inventory(*, base_dir: Path | None = None) -> SeedInventory:
    resolved_base_dir = _resolve_workspace_root(base_dir)
    provider_repos = discover_provider_repos(resolved_base_dir)
    discovered_roots = _discover_repo_artifact_roots(resolved_base_dir)
    parsed_files = 0
    parsed_atoms = 0
    duplicate_fqdns: list[str] = []
    empty_roots: list[str] = []
    repository_rows: list[RepositorySeedRow] = []
    atom_rows_by_fqdn: dict[str, AtomSeedRow] = {}
    version_rows_by_fqdn: dict[str, VersionSeedRow] = {}
    repo_row_names: set[str] = set()

    for priority, repo, artifact_root in discovered_roots:
        py_files = tuple(_iter_python_files(artifact_root))
        if not py_files:
            empty_roots.append(str(artifact_root))
            continue
        parsed_files += len(py_files)
        namespace_root = ".".join(namespace_prefix_for_artifact_root(artifact_root))
        if repo.repo_name not in repo_row_names:
            repo_url = _git_remote_url(repo.repo_root)
            repository_rows.append(
                RepositorySeedRow(
                    repo_name=repo.repo_name,
                    repo_url=repo_url,
                    namespace_root=namespace_root,
                    clone_priority=priority,
                    vcs_provider=_vcs_provider(repo_url),
                )
            )
            repo_row_names.add(repo.repo_name)

        for parsed in _parse_registered_atoms(repo=repo, artifact_root=artifact_root):
            parsed_atoms += 1
            atom_row = AtomSeedRow(
                fqdn=parsed.fqdn,
                namespace_root=parsed.namespace_root,
                namespace_path=parsed.namespace_path,
                repo_name=parsed.repo_name,
                source_package=parsed.namespace_root,
                source_module_path=parsed.source_module_path,
                source_symbol=parsed.source_symbol,
                description=parsed.description,
                domain_tags=parsed.domain_tags,
                source_kind=parsed.source_kind,
                is_ffi=parsed.is_ffi,
            )
            if parsed.fqdn in atom_rows_by_fqdn:
                duplicate_fqdns.append(parsed.fqdn)
                continue
            atom_rows_by_fqdn[parsed.fqdn] = atom_row
            version_rows_by_fqdn[parsed.fqdn] = VersionSeedRow(
                fqdn=parsed.fqdn,
                version_id=parsed.version_id,
                content_hash=parsed.content_hash,
                semver=parsed.semver,
                s3_key=parsed.s3_key,
                fingerprint=parsed.fingerprint,
            )

    repository_rows.sort(key=lambda row: (row.clone_priority, row.repo_name))
    atom_rows = tuple(sorted(atom_rows_by_fqdn.values(), key=lambda row: row.fqdn))
    version_rows = tuple(version_rows_by_fqdn[key] for key in sorted(version_rows_by_fqdn))
    hyperparam_rows = _derive_hyperparam_rows(resolved_base_dir)
    benchmark_suite_rows, benchmark_result_rows = _derive_benchmark_rows(resolved_base_dir)
    return SeedInventory(
        base_dir=resolved_base_dir,
        provider_repos=tuple(provider_repos),
        repository_rows=tuple(repository_rows),
        atom_rows=atom_rows,
        version_rows=version_rows,
        hyperparam_rows=hyperparam_rows,
        benchmark_suite_rows=benchmark_suite_rows,
        benchmark_result_rows=benchmark_result_rows,
        parsed_atoms=parsed_atoms,
        parsed_files=parsed_files,
        duplicate_fqdns=_dedupe_preserve_order(duplicate_fqdns),
        empty_roots=_dedupe_preserve_order(empty_roots),
    )


def build_repository_rows(inventory: SeedInventory) -> list[dict[str, Any]]:
    return [row.as_dict() for row in inventory.repository_rows]


def build_atom_rows(
    inventory: SeedInventory,
    *,
    owner_id: str,
    source_repo_ids: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    source_repo_ids = source_repo_ids or {}
    return [
        row.as_dict(owner_id=owner_id, source_repo_id=source_repo_ids.get(row.repo_name))
        for row in inventory.atom_rows
    ]


def build_version_rows(
    inventory: SeedInventory,
    *,
    atom_ids: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    skipped_no_atom = 0
    touched_atom_ids: set[str] = set()
    for spec in inventory.version_rows:
        atom_id = atom_ids.get(spec.fqdn)
        if atom_id is None:
            skipped_no_atom += 1
            continue
        touched_atom_ids.add(atom_id)
        rows.append(spec.as_dict(atom_id=atom_id))
    return rows, {
        "version_rows": len(rows),
        "version_atoms": len(touched_atom_ids),
        "version_skipped_no_atom": skipped_no_atom,
    }


def build_hyperparam_rows(
    inventory: SeedInventory,
    *,
    atom_ids: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    skipped_no_atom = 0
    touched_atom_ids: set[str] = set()
    for spec in inventory.hyperparam_rows:
        atom_id = atom_ids.get(spec.fqdn)
        if atom_id is None:
            skipped_no_atom += 1
            continue
        touched_atom_ids.add(atom_id)
        rows.append(spec.as_dict(atom_id=atom_id))
    return rows, {
        "hyperparam_rows": len(rows),
        "hyperparam_atoms": len(touched_atom_ids),
        "hyperparam_skipped_no_atom": skipped_no_atom,
    }


def build_benchmark_suite_rows(inventory: SeedInventory) -> list[dict[str, Any]]:
    return [row.as_dict() for row in inventory.benchmark_suite_rows]


def build_atom_benchmark_rows(
    inventory: SeedInventory,
    *,
    version_ids: dict[tuple[str, str], str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    deferred_cdg_results = 0
    skipped_no_version = 0
    touched_versions: set[str] = set()
    for result in inventory.benchmark_result_rows:
        if result.artifact_kind != "atom":
            deferred_cdg_results += 1
            continue
        version_id = version_ids.get((result.artifact_fqdn, result.content_hash))
        if version_id is None:
            skipped_no_version += 1
            continue
        touched_versions.add(version_id)
        rows.append(result.as_atom_benchmark_dict(version_id=version_id))
    return rows, {
        "benchmark_suite_rows": len(inventory.benchmark_suite_rows),
        "benchmark_result_rows": len(inventory.benchmark_result_rows),
        "atom_benchmark_rows": len(rows),
        "benchmark_atom_versions": len(touched_versions),
        "benchmark_result_cdg_seen": deferred_cdg_results,
        "benchmark_result_skipped_no_version": skipped_no_version,
    }


def build_artifact_benchmark_rows(
    inventory: SeedInventory,
    *,
    version_ids: dict[tuple[str, str], str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    skipped_no_version = 0
    touched_versions: set[str] = set()
    for result in inventory.benchmark_result_rows:
        if result.artifact_kind != "cdg":
            continue
        version_id = version_ids.get((result.artifact_fqdn, result.content_hash))
        if version_id is None:
            skipped_no_version += 1
            continue
        touched_versions.add(version_id)
        rows.append(result.as_atom_benchmark_dict(version_id=version_id))
    return rows, {
        "artifact_benchmark_rows": len(rows),
        "benchmark_artifact_versions": len(touched_versions),
        "benchmark_cdg_skipped_no_version": skipped_no_version,
    }


def build_owner_seed(owner_login: str = DEFAULT_OWNER_LOGIN, *, email: str | None = None) -> OwnerSeed:
    normalized_login = (owner_login or DEFAULT_OWNER_LOGIN).strip() or DEFAULT_OWNER_LOGIN
    user_id = str(uuid5(NAMESPACE_URL, f"sciona-seed-owner:{normalized_login}"))
    return OwnerSeed(
        user_id=user_id,
        github_id=DEFAULT_OWNER_GITHUB_ID,
        github_login=normalized_login,
        display_name=DEFAULT_OWNER_DISPLAY_NAME,
        avatar_url="",
        email=(email or f"{normalized_login}@localhost").strip() or DEFAULT_OWNER_EMAIL,
    )


def render_owner_seed_sql(owner: OwnerSeed) -> str:
    safe_login = owner.github_login.replace("'", "''")
    safe_name = owner.display_name.replace("'", "''")
    safe_avatar = owner.avatar_url.replace("'", "''")
    safe_email = owner.email.replace("'", "''")
    return "\n".join(
        [
            "-- deterministic local owner bootstrap",
            "INSERT INTO auth.users (",
            "  id, aud, role, email, email_confirmed_at, raw_app_meta_data, raw_user_meta_data, created_at, updated_at",
            ") VALUES (",
            f"  '{owner.user_id}'::uuid,",
            "  'authenticated',",
            "  'authenticated',",
            f"  '{safe_email}',",
            "  now(),",
            "  '{\"provider\":\"email\",\"providers\":[\"email\"]}'::jsonb,",
            f"  jsonb_build_object('user_name', '{safe_login}', 'full_name', '{safe_name}', 'avatar_url', '{safe_avatar}'),",
            "  now(),",
            "  now()",
            ")",
            "ON CONFLICT (id) DO NOTHING;",
            "",
            "INSERT INTO public.users (",
            "  user_id, github_id, github_login, display_name, avatar_url, email,",
            "  identity_tier, reputation_score, is_blacklisted, effective_tier",
            ") VALUES (",
            f"  '{owner.user_id}'::uuid,",
            f"  {owner.github_id},",
            f"  '{safe_login}',",
            f"  '{safe_name}',",
            f"  '{safe_avatar}',",
            f"  '{safe_email}',",
            "  'contributor',",
            "  0,",
            "  FALSE,",
            "  'general'",
            ")",
            "ON CONFLICT (user_id) DO UPDATE SET",
            "  github_login = EXCLUDED.github_login,",
            "  display_name = EXCLUDED.display_name,",
            "  avatar_url = EXCLUDED.avatar_url,",
            "  email = EXCLUDED.email,",
            "  updated_at = now();",
        ]
    )


async def _ensure_owner_via_database_url(database_url: str, owner: OwnerSeed) -> None:
    import asyncpg

    conn = await asyncpg.connect(database_url)
    try:
        raw_app_meta_data = {"provider": "email", "providers": ["email"]}
        raw_user_meta_data = {
            "user_name": owner.github_login,
            "full_name": owner.display_name,
            "avatar_url": owner.avatar_url,
        }
        await conn.execute(
            """
            INSERT INTO auth.users (
                id,
                aud,
                role,
                email,
                email_confirmed_at,
                raw_app_meta_data,
                raw_user_meta_data,
                created_at,
                updated_at
            )
            VALUES (
                $1::uuid,
                'authenticated',
                'authenticated',
                $2,
                now(),
                $3::jsonb,
                $4::jsonb,
                now(),
                now()
            )
            ON CONFLICT (id) DO NOTHING
            """,
            owner.user_id,
            owner.email,
            json.dumps(raw_app_meta_data),
            json.dumps(raw_user_meta_data),
        )
        await conn.execute(
            """
            INSERT INTO public.users (
                user_id,
                github_id,
                github_login,
                display_name,
                avatar_url,
                email,
                identity_tier,
                reputation_score,
                is_blacklisted,
                effective_tier
            )
            VALUES (
                $1::uuid, $2, $3, $4, $5, $6,
                'contributor', 0, FALSE, 'general'
            )
            ON CONFLICT (user_id) DO UPDATE SET
                github_login = EXCLUDED.github_login,
                display_name = EXCLUDED.display_name,
                avatar_url = EXCLUDED.avatar_url,
                email = EXCLUDED.email,
                updated_at = now()
            """,
            owner.user_id,
            owner.github_id,
            owner.github_login,
            owner.display_name,
            owner.avatar_url,
            owner.email,
        )
    finally:
        await conn.close()


def ensure_owner_via_database_url(database_url: str, owner: OwnerSeed) -> None:
    asyncio.run(_ensure_owner_via_database_url(database_url, owner))


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


def _upsert_rows(client: Any, table: str, rows: Sequence[dict[str, Any]], *, conflict: str) -> Any:
    if not rows:
        return None
    return client.table(table).upsert(list(rows), on_conflict=conflict).execute()


def _replace_atom_benchmark_rows(client: Any, rows: Sequence[dict[str, Any]]) -> None:
    touched_pairs = sorted({(str(row["version_id"]), str(row["benchmark_name"])) for row in rows})
    for version_id, benchmark_name in touched_pairs:
        client.table("atom_benchmarks").delete().eq("version_id", version_id).eq("benchmark_name", benchmark_name).execute()
    if rows:
        client.table("atom_benchmarks").insert(list(rows)).execute()


def _replace_artifact_benchmark_rows(client: Any, rows: Sequence[dict[str, Any]]) -> None:
    touched_pairs = sorted({(str(row["version_id"]), str(row["benchmark_name"])) for row in rows})
    for version_id, benchmark_name in touched_pairs:
        client.table("artifact_benchmarks").delete().eq("version_id", version_id).eq("benchmark_name", benchmark_name).execute()
    if rows:
        client.table("artifact_benchmarks").insert(list(rows)).execute()


def _fetch_source_repo_ids(client: Any) -> dict[str, str]:
    response = client.table("atom_source_repositories").select("repo_name,source_repo_id").execute()
    return {
        str(row["repo_name"]): str(row["source_repo_id"])
        for row in (getattr(response, "data", None) or [])
        if row.get("repo_name") and row.get("source_repo_id")
    }


def _fetch_atom_ids(client: Any) -> dict[str, str]:
    response = client.table("atoms").select("fqdn,atom_id").execute()
    return {
        str(row["fqdn"]): str(row["atom_id"])
        for row in (getattr(response, "data", None) or [])
        if row.get("fqdn") and row.get("atom_id")
    }


def _fetch_artifact_version_ids(client: Any) -> dict[tuple[str, str], str]:
    artifact_rows = getattr(
        client.table("artifact_versions").select("version_id,content_hash,artifact_id").execute(),
        "data",
        None,
    ) or []
    artifact_ids = {str(row["artifact_id"]) for row in artifact_rows if row.get("artifact_id")}
    if not artifact_ids:
        return {}
    fqdn_rows = getattr(
        client.table("artifacts").select("artifact_id,fqdn").in_("artifact_id", sorted(artifact_ids)).execute(),
        "data",
        None,
    ) or []
    fqdn_by_id = {str(row["artifact_id"]): str(row["fqdn"]) for row in fqdn_rows if row.get("artifact_id") and row.get("fqdn")}
    return {
        (fqdn_by_id[str(row["artifact_id"])], str(row["content_hash"])): str(row["version_id"])
        for row in artifact_rows
        if row.get("artifact_id") and row.get("content_hash") and row.get("version_id")
        and str(row["artifact_id"]) in fqdn_by_id
    }


def seed_core_supabase(
    client: Any,
    inventory: SeedInventory | None = None,
    *,
    base_dir: Path | None = None,
    owner: OwnerSeed | None = None,
    ensure_owner: bool = False,
    database_url: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    inventory = inventory or derive_seed_inventory(base_dir=base_dir)
    owner = owner or build_owner_seed()
    summary = inventory.summary()
    summary["owner"] = {
        "user_id": owner.user_id,
        "github_login": owner.github_login,
        "email": owner.email,
    }
    summary["dry_run"] = dry_run
    if dry_run:
        return summary

    if ensure_owner:
        resolved_database_url = (
            database_url
            or os.environ.get("SUPABASE_DATABASE_URL")
            or os.environ.get("SCIONA_SUPABASE_DATABASE_URL")
        )
        if not resolved_database_url:
            raise RuntimeError("SUPABASE_DATABASE_URL is required when --ensure-owner is set")
        ensure_owner_via_database_url(resolved_database_url, owner)

    _upsert_rows(
        client,
        "atom_source_repositories",
        build_repository_rows(inventory),
        conflict="repo_name",
    )
    _upsert_rows(
        client,
        "benchmark_suites",
        build_benchmark_suite_rows(inventory),
        conflict="benchmark_id",
    )
    repo_ids = _fetch_source_repo_ids(client)
    _upsert_rows(
        client,
        "atoms",
        build_atom_rows(inventory, owner_id=owner.user_id, source_repo_ids=repo_ids),
        conflict="fqdn",
    )
    atom_ids = _fetch_atom_ids(client)
    version_rows, version_summary = build_version_rows(inventory, atom_ids=atom_ids)
    touched_version_atom_ids = sorted({row["atom_id"] for row in version_rows})
    for atom_id in touched_version_atom_ids:
        client.table("atom_versions").update({"is_latest": False}).eq("atom_id", atom_id).execute()
        client.table("atom_versions").delete().eq("atom_id", atom_id).like("s3_key", "local-seed/%").execute()
    _upsert_rows(
        client,
        "atom_versions",
        version_rows,
        conflict="version_id",
    )
    hyperparam_rows, hyperparam_summary = build_hyperparam_rows(inventory, atom_ids=atom_ids)
    for atom_id in sorted({row["atom_id"] for row in hyperparam_rows}):
        client.table("hyperparams").delete().eq("atom_id", atom_id).execute()
    _upsert_rows(
        client,
        "hyperparams",
        hyperparam_rows,
        conflict="atom_id,name",
    )
    atom_version_ids = {(row.fqdn, row.content_hash): row.version_id for row in inventory.version_rows}
    atom_benchmark_rows, benchmark_summary = build_atom_benchmark_rows(inventory, version_ids=atom_version_ids)
    _replace_atom_benchmark_rows(client, atom_benchmark_rows)
    artifact_version_ids = _fetch_artifact_version_ids(client)
    artifact_benchmark_rows, artifact_benchmark_summary = build_artifact_benchmark_rows(
        inventory,
        version_ids=artifact_version_ids,
    )
    _replace_artifact_benchmark_rows(client, artifact_benchmark_rows)
    summary["repo_ids"] = repo_ids
    summary.update(version_summary)
    summary.update(hyperparam_summary)
    summary.update(benchmark_summary)
    summary.update(artifact_benchmark_summary)
    summary["applied"] = True
    return summary


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seed provider-owned Supabase repository and atom rows."
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help="Workspace root containing sibling provider repos.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write to Supabase instead of only printing a dry-run summary.",
    )
    parser.add_argument(
        "--ensure-owner",
        action="store_true",
        help="Insert the deterministic local owner into auth.users/public.users first.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Optional direct Postgres URL for owner bootstrapping.",
    )
    parser.add_argument(
        "--owner-login",
        default=DEFAULT_OWNER_LOGIN,
        help="Deterministic local owner login.",
    )
    parser.add_argument(
        "--owner-email",
        default=None,
        help="Optional local owner email.",
    )
    parser.add_argument(
        "--print-owner-sql",
        action="store_true",
        help="Print the deterministic owner bootstrap SQL and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output instead of a compact summary line.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    owner = build_owner_seed(args.owner_login, email=args.owner_email)
    if args.print_owner_sql:
        print(render_owner_seed_sql(owner))
        return 0

    inventory = derive_seed_inventory(base_dir=args.workspace_root)
    if args.apply:
        summary = seed_core_supabase(
            create_supabase_client_from_env(),
            inventory=inventory,
            owner=owner,
            ensure_owner=args.ensure_owner,
            database_url=args.database_url,
            dry_run=False,
        )
    else:
        summary = inventory.summary()
        summary["owner"] = {
            "user_id": owner.user_id,
            "github_login": owner.github_login,
            "email": owner.email,
        }
        summary["dry_run"] = True

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"repos={summary['repository_rows']} atoms={summary['atom_rows']} "
            f"parsed_atoms={summary['parsed_atoms']} dry_run={summary['dry_run']}"
        )
    return 0


def main() -> int:
    import sys

    return run_cli(sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
