# Atom Ingestion Guide

This is the shared ingestion contract for the sciona atom ecosystem. It lives in
`sciona-atoms` and is referenced by all sibling atom repositories.

It covers:

1. the ecosystem topology and how sibling repos relate
2. key concepts and terminology
3. how `sciona ingest` emits atoms into any sibling repo
4. what must be true before an ingest is considered durable, reviewable, and
   acceptable

Post-ingestion standards for interfaces, documentation, probes, hyperparameters,
heuristics, and review are defined in
[CONTRIBUTION.md](../sciona-atoms/CONTRIBUTION.md). This document does not
repeat them — it covers the ingestion workflow, artifact handling, and audit
sequencing that gets an atom to the point where those standards apply.

## Core Principle

Ingestion is not complete because the matcher CLI succeeded. It is complete only
when the stable artifacts, package shape, provenance, documentation, runtime
evidence, and deterministic audit state are all coherent.

## Glossary

These terms appear throughout the ingestion and contribution contracts:

- **atom** — A stateless, typed, contracted wrapper function that exposes a
  single unit of computation. Atoms are the fundamental building block of the
  sciona system. Each atom is registered with `@register_atom(...)` and guarded
  by `@icontract.require`/`@icontract.ensure` preconditions and postconditions.

- **witness** — A pure, side-effect-free function that mirrors an atom's
  interface using abstract ghost types. Witnesses describe the semantic shape of
  an atom (input/output types, state propagation, dtype/shape contracts) without
  executing real computation. They live in `witnesses.py` alongside their atoms.

- **CDG (Conceptual Dependency Graph)** — A JSON structure (`cdg.json`) that
  describes how a set of atoms decompose a larger upstream algorithm. Nodes
  represent atoms; edges represent data and state dependencies.

- **ghost simulation** — A verification pass that executes the CDG symbolically
  using witness definitions, checking for logical inconsistencies, cyclic
  dependencies, and coverage gaps without running real computation.

- **matcher** — The `sciona-matcher` repository and its `sciona ingest` CLI.
  The matcher is the code-generation and ingestion-time verification engine.

- **matcher grounding** — The process by which the matcher maps generated atoms
  to known library functions. Results are stored in `matches.json`.

- **probe** — A deterministic, bounded verification check defined in the atom
  repo. Probes provide runtime evidence that an atom behaves as contracted.
  They live under `src/sciona/probes/`.

- **family** — A group of related atoms that were decomposed from the same
  upstream source and share a common output directory.

- **state model** — A structured representation (typically a Pydantic model or
  dataclass) of the durable cross-call state for stateful atom families. Lives
  in `state_models.py` when present.

## Ecosystem Topology

### Sibling Repository Layout

All sciona repositories share the `sciona` Python namespace via PEP 420
implicit namespace packages. Every sibling repo uses the same
`src/sciona/...` source layout and contributes subpackages to the shared
namespace.

```text
sciona-matcher/       # owns `sciona ingest`, the ingestion CLI and venv
sciona-atoms/         # shared infrastructure: scripts, validation, CONTRIBUTION.md, INGESTION.md
sciona-atoms-cs/      # example domain-specific sibling atom repo
sciona-atoms-xyz/     # another domain-specific sibling atom repo
```

Key relationships:

- `sciona-matcher` owns the ingestion CLI and the shared Python virtualenv
- `sciona-atoms` owns the shared validation scripts, contribution contract, and
  ingestion contract (this file)
- Domain-specific sibling repos (e.g., `sciona-atoms-cs`) contain atoms for
  their domain and depend on both `sciona-matcher` and `sciona-atoms` as
  siblings

All sibling repos contribute to the same `sciona` namespace. Atoms live under
`src/sciona/atoms/`, probes under `src/sciona/probes/`, regardless of which
sibling repo they are in. The PEP 420 namespace mechanism ensures that
`import sciona.atoms.numpy` and `import sciona.atoms.cs.some_family` resolve
to the correct sibling repo transparently.

### Namespace Compatibility

The atoms repo currently uses a `pkgutil.extend_path` shim in
`src/sciona/__init__.py` for compatibility while the matcher still owns a
classic `sciona` package. Sibling repos should use the same shim pattern. Once
the matcher converts to namespace-compatible packaging, these shims will be
removed.

### Running Commands

- Run `sciona ingest` from `../sciona-matcher`
- Run Python and tests with `../sciona-matcher/.venv/bin/python`
- Run validation and audit scripts from `../sciona-atoms`

## `sciona ingest` Usage

Run ingestion from `../sciona-matcher`, targeting the atom's final location in
your repo:

```bash
cd ../sciona-matcher
sciona ingest <source_file> --class <ClassName> \
  --output ../<your-repo>/src/sciona/atoms/<domain>/<atom_name>
```

Note: `--class` is used for both classes and standalone functions — it takes the
name of the symbol to ingest regardless of whether it is a class or a function.

### Examples

**Ingesting a standalone function:**

```bash
cd ../sciona-matcher
sciona ingest ../scipy/scipy/signal/windows/_windows.py \
  --class kaiser \
  --procedural \
  --output ../sciona-atoms/src/sciona/atoms/scipy/signal/kaiser
```

Use `--procedural` for standalone functions. This uses deterministic SSA edge
inference with no LLM calls — the function is parsed, variable dependencies are
tracked statically, and the atom wrapper is emitted directly.

**Ingesting a stateful class:**

```bash
cd ../sciona-matcher
sciona ingest ../scikit-learn/sklearn/decomposition/_pca.py \
  --class PCA \
  --output ../sciona-atoms-cs/src/sciona/atoms/decomposition/pca
```

Without `--procedural`, the matcher uses LLM-driven semantic decomposition. It
extracts methods, state attributes, and cross-method dependencies, then
decomposes the class into a family of atoms with a shared `state_models.py`.
The emitted `cdg.json` captures the dependency structure between the resulting
atoms.

### Common Options

| Flag | Purpose |
|------|---------|
| `--procedural` | Deterministic SSA extraction, bypasses LLM |
| `--llm-provider <provider>` | Override LLM provider |
| `--llm-model <model>` | Override LLM model |
| `--trace` | Write pipeline event trace to `trace.jsonl` |
| `--monitor` | Print live ingestion status to stdout |
| `--output-scope family` | Grouped output for multi-symbol families |
| `--allow-family-replace` | Replace entire family output |
| `--allow-family-merge` | Merge new artifacts without replacing untouched files |

### Ingestion Pipeline Phases

The matcher runs a 5-phase pipeline with verification and repair loops:

1. **Extract** — Deterministic AST/tree-sitter parsing into a `RawDataFlowGraph`
2. **Chunk** — LLM-driven semantic decomposition into a `ValidatedMacroPlan`
3. **Emit** — Code generation of `atoms.py`, `witnesses.py`, `state_models.py`,
   and `cdg.json`
4. **Verify Types** — mypy strict-mode type checking with up to 3 repair cycles
5. **Verify Ghost Simulation** — Ghost simulation on the CDG with repair loops
   for logical inconsistencies and cyclic dependencies

Ingest-time smoke validation runs before artifacts are published. A passing
smoke check is necessary but not sufficient — repo-side audit is the acceptance
layer.

### Output Scope Rules

- **Symbol scope** (default): Each ingestion targets a dedicated output
  directory for that symbol
- **Family scope**: Multiple related ingestions share one output directory;
  requires explicit `--allow-family-replace` or `--allow-family-merge`

Ingest directly to the final repo location when feasible. Do not treat a scratch
tree as the committed artifact. If grouped output is intended, make that explicit
rather than accepting accidental one-symbol sprawl or accidental grouped
overwrite.

## Default Output Layout

New atoms use the per-atom directory layout:

```text
src/sciona/atoms/
  <domain>/
    __init__.py
    <atom_name>/
      __init__.py
      atoms.py
      witnesses.py
      state_models.py      # only when state is real and durable
      cdg.json
      matches.json         # when matcher grounding exists
      references.json      # when scholarly/provenance metadata exists
      uncertainty.json     # when perturbation analysis has been run
```

This layout is the same in every sibling repo. The `src/sciona/atoms/` prefix
is shared across all repos via PEP 420 namespacing — the `<domain>/` level is
what distinguishes atoms between repos.

Legacy flat layouts (e.g., `<domain>/<module>.py` with sibling
`<stem>_witnesses.py` and `<stem>_cdg.json`) remain valid for existing atoms.
Do not churn legacy atoms into the new layout unless the work genuinely requires
it.

## Stable Artifacts vs Ephemeral Byproducts

### Commit These (Durable Artifacts)

- `atoms.py`
- `witnesses.py`
- `state_models.py` when the atom is genuinely stateful
- `cdg.json` (or legacy `*_cdg.json`)
- `matches.json` when matcher grounding produced it
- `references.json` when attribution/provenance exists
- `uncertainty.json` when uncertainty was measured
- relevant `__init__.py` updates

### Do Not Commit (Operational Byproducts)

- `.ingest_status.json`
- `COMPLETED.json`
- `FAILED.json`
- `trace.jsonl`
- `shared_context_metrics.json`
- local `logs/`
- local `.playwright-mcp/`

## Post-Ingest Fixup

The matcher output is a starting point. Before the atom is contribution-ready,
fix file-scoped issues immediately:

- Signature/default drift from upstream semantics
- Weak type surfaces (`Any`, untyped generics)
- Local CDG issues (duplicate anonymous nodes, fake orchestration)
- Witness drift (wrong ghost types, conceptually unrelated abstractions)
- Naming/alignment drift
- Missing or incoherent package exports

The interface, witness, CDG, and documentation standards that apply after fixup
are defined in [CONTRIBUTION.md](../sciona-atoms/CONTRIBUTION.md).

## Provenance And Scholarly References

When authoritative upstream sources exist, ingestion work is not done until
provenance is reviewable.

### Upstream Mapping

Use `scripts/atom_manifest.yml` in your repo (when present) to record:

- exact upstream repo/module/function anchors
- curated closest anchors for refined-ingest adapters
- notes when the wrapper decomposes a larger upstream algorithm

Rules:

- prefer exact anchors over vague module-level notes
- do not use a single ambiguous manifest entry for multiple exported helpers
- keep overloaded or decomposed helpers disambiguated at the atom level

### Scholarly References

Use per-atom `references.json` for papers, books, standards, and other
scholarly material.

Rules:

- prefer DOI-backed entries when possible
- keep `references.json` stable once curated
- preserve manual match metadata and notes

## Uncertainty

`uncertainty.json` is a durable post-ingest artifact when empirical perturbation
analysis has been run.

Rules:

- do not fabricate uncertainty files manually
- preserve existing `uncertainty.json`
- if an atom is a strong uncertainty candidate and the work already touches the
  family, consider measuring uncertainty in the same tranche

## Matcher Grounding

`matches.json` is a durable matcher handoff artifact. Keep it when matcher
grounding emitted it — it records grounding output not represented elsewhere and
supports later synthesis and audit work.

## Deterministic Audit Workflow

Matcher ingest success is not acceptance. The deterministic audit stack in
`sciona-atoms` is the acceptance layer. All sibling repos run these scripts
from `../sciona-atoms/`.

### Required Audit Order

Run from your repo's root after meaningful ingest or remediation work. The
scripts live in `../sciona-atoms/scripts/` and are executed with the matcher's
venv:

```bash
../sciona-matcher/.venv/bin/python ../sciona-atoms/scripts/build_audit_manifest.py
../sciona-matcher/.venv/bin/python ../sciona-atoms/scripts/validate_audit_manifest.py
../sciona-matcher/.venv/bin/python ../sciona-atoms/scripts/audit_structural.py
../sciona-matcher/.venv/bin/python ../sciona-atoms/scripts/audit_signature_fidelity.py
env PYTHON_JULIAPKG_PROJECT=/tmp/sciona_juliapkg_project \
  JULIA_DEPOT_PATH=/tmp/sciona_julia_depot \
  MPLCONFIGDIR=/tmp/mpl \
  ../sciona-matcher/.venv/bin/python ../sciona-atoms/scripts/audit_runtime_probes.py
../sciona-matcher/.venv/bin/python ../sciona-atoms/scripts/report_parity_coverage.py
../sciona-matcher/.venv/bin/python ../sciona-atoms/scripts/audit_return_fidelity.py
../sciona-matcher/.venv/bin/python ../sciona-atoms/scripts/audit_state_fidelity.py
../sciona-matcher/.venv/bin/python ../sciona-atoms/scripts/audit_generated_nouns.py
../sciona-matcher/.venv/bin/python ../sciona-atoms/scripts/audit_semantics.py
../sciona-matcher/.venv/bin/python ../sciona-atoms/scripts/audit_acceptability.py
../sciona-matcher/.venv/bin/python ../sciona-atoms/scripts/audit_risk.py
```

### Sequential Tail Constraint

These stages rewrite shared manifest state and must remain sequential:

- `report_parity_coverage.py`
- `audit_semantics.py`
- `audit_acceptability.py`
- `audit_risk.py`

Do not parallelize them.

### Focused Family Work

For family-local work, run focused `pytest` slices first, then the full ordered
audit stack before declaring the repo truthful.

## Validation Gate

Before opening a PR, run the shared verification scripts from
`../sciona-atoms/` (see [CONTRIBUTION.md](../sciona-atoms/CONTRIBUTION.md) for
full standards):

```bash
../sciona-matcher/.venv/bin/python ../sciona-atoms/scripts/verify_contribution_rules.py --repo-root .
../sciona-matcher/.venv/bin/python ../sciona-atoms/scripts/validate_dejargon.py --root .
```

These are structural and deterministic. They are the merge gate, not the full
audit.

## Review And Trust Rules

Low risk is not the same thing as trusted.

- Deterministic evidence can reduce risk and improve acceptability
- Review basis is a separate trust layer
- Do not claim an atom is trusted just because it is low risk
- Preserve and respect review workflow artifacts under `data/audit_reviews`
  when present in your repo

## Agent Procedure

1. Choose the final output path under `src/sciona/atoms/` in your repo.
2. Run `sciona ingest` from `../sciona-matcher`.
3. Keep durable artifacts and discard monitor byproducts.
4. Fix local wrapper/witness/CDG issues immediately when they are file-scoped.
5. Update exports, manifest mappings, references, and uncertainty artifacts as
   needed.
6. Add or update runtime probes and focused tests when the family needs parity
   evidence.
7. Rerun the deterministic audit stack from `../sciona-atoms/scripts/` in order.
8. Inspect the resulting manifest, semantic status, acceptability, and risk.
9. Run the validation gate from `../sciona-atoms/scripts/`.
10. Do not stop until the repo is left in a truthful, reviewable state.

## Acceptance Checklist

```text
Artifacts:
  [ ] Stable files are in the correct src/sciona/atoms/ location in your repo
  [ ] No ingest monitor sidecars are committed
  [ ] Package exports are updated coherently

Interface (see ../sciona-atoms/CONTRIBUTION.md for full standards):
  [ ] Public atoms are typed, registered, and honestly contracted
  [ ] Witnesses use valid abstract types only
  [ ] Stateful wrappers use explicit documented state
  [ ] CDG matches the actual wrapper decomposition

Metadata:
  [ ] matches.json kept when matcher grounding exists
  [ ] references.json preserved or added when authoritative sources exist
  [ ] atom_manifest.yml mapping updated when provenance matters
  [ ] uncertainty.json preserved or generated when appropriate

Audit:
  [ ] Audit manifest rebuilt and validated
  [ ] Structural/fidelity/probe audits rerun
  [ ] Semantic/acceptability/risk tail rerun sequentially
  [ ] Atom is not left broken, misleading, or falsely confident

Validation:
  [ ] verify_contribution_rules.py passes
  [ ] validate_dejargon.py passes
  [ ] Relevant tests pass

Trust:
  [ ] Risk state is acceptable for the intended change
  [ ] Review-basis needs are not hand-waved away
```
