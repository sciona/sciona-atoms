# Agent-Driven Atom Ingestion Guide

This guide teaches CLI agents (Claude Code, Codex, etc.) how to ingest atoms
into the sciona ecosystem using deterministic tools, with a human supervising
the process.

The agent reads source code, decides how to decompose it into atoms, writes
the atom/witness/CDG artifacts, runs verification, and iterates until clean.
The human directs the work and makes judgment calls when the agent is uncertain.

## Core Principle: Ask, Don't Guess

Any decision that could poison the atom registry must be escalated to the human
rather than hallucinated. A bad atom that gets matched repeatedly is worse than
no atom — it wastes tokens, CPU time, and erodes trust in the catalog.

When uncertain about decomposition boundaries, naming, licensing, or whether a
fix is correct — **stop and ask the human**.

## Environment

```
Python venv:      ../sciona-matcher/.venv/bin/python
Matcher repo:     ../sciona-matcher/
Atoms repo:       ../sciona-atoms/          (this repo — shared contracts)
Domain repos:     ../sciona-atoms-{bio,cs,fintech,ml,physics,robotics,signal}/
Tools import:     from sciona.tools import ...
Validation:       ../sciona-atoms/scripts/verify_contribution_rules.py
                  ../sciona-atoms/scripts/validate_dejargon.py
```

All tool imports assume the sciona-matcher venv is active. If running from a
different environment, activate the venv first:

```bash
source ../sciona-matcher/.venv/bin/activate
```

## Available Tools

Import from `sciona.tools`. Each function has detailed docstrings — use
`help(fn)` for full signatures and return types.

| Tool | Purpose | Phase |
|------|---------|-------|
| `extract_dfg` | Data-flow graph for a Python class | 1 |
| `extract_function_dfg` | Data-flow graph for a standalone function | 1 |
| `extract_procedural_dfg` | SSA-based extraction for scripts | 1 |
| `run_mypy` | Run mypy --strict on source files dict | 6 |
| `classify_type_failure` | Classify mypy errors as repairable or semantic | 6 |
| `build_type_fixes` | Deterministic patches for mypy errors | 6 |
| `classify_ghost_failure` | Classify ghost sim failure | 6 |
| `build_ghost_fixes` | Deterministic patches for ghost failures | 6 |
| `detect_cycles` | Kahn's algorithm cycle detection on CDG | 5 |
| `break_cycle` | Deterministic cycle-breaking patches | 5 |
| `validate_cdg_ir` | Validate CDG covers all attributes | 5 |
| `match_witness_template` | Template match for DL layer witnesses | 4 |
| `generate_abstract_profile` | Cross-domain conceptual profile | 5 |
| `run_contribution_check` | Run contribution rules validator | 6 |
| `run_dejargon_check` | Run dejargon validator | 6 |

## Workflow

### Phase 1: Read and Understand

Use `extract_dfg` or `extract_function_dfg` for structural facts. Also read
the source code directly — the DFG is structural, but the code reveals intent,
naming conventions, and domain semantics.

Key questions:
- What methods exist? What are their semantic roles?
- What state is fitted vs config vs transient?
- What helpers does the function depend on?
- Are there unknowns or ambiguities?

### Phase 2: Decide on Decomposition

- **Which atoms?** Each method with a clear, separable role becomes one atom.
  Helpers called by only one public method should be inlined, not separate.
- **Which family?** Group atoms that share state, imports, or conceptual scope
  into one family directory with a shared `atoms.py` and `witnesses.py`.
- **State model needed?** If cross-window state exists, you need `state_models.py`.
- **Uncertain?** Ask the human.

### Phase 3: Write atoms.py

Every public atom must:

- Use `@register_atom(witness_fn)` as the outermost decorator
- Include at least one `@icontract.require` precondition (meaningful, not
  tautological)
- **Include at least one `@icontract.ensure` postcondition** — enforced by
  `verify_contribution_rules.py`. Common postconditions: `np.isfinite(result)`,
  `result >= 0`, `len(result) > 0`, `0 <= result <= 1`.
- Be fully type-annotated — no `Any` in public interfaces unless justified
- Have an honest docstring describing what it computes, not boilerplate

### Phase 4: Write witnesses.py

Witnesses are pure, side-effect-free functions that mirror the atom interface
using abstract ghost types.

- Use `AbstractArray` for tensor parameters and return values
- Capture shape transforms symbolically
- Keep witnesses minimal — no real computation
- For DL layers, try `match_witness_template` first

### Phase 5: Write cdg.json

The CDG captures dependency structure between atoms. **Every atomic node must
have concrete `inputs` and `outputs`** — without these, the IO backfill
pipeline cannot generate `atom_io_specs` rows and the atom will not be
publishable.

Rules:
- The `name` field on each atomic node must be the snake_case function name
  (e.g., `"igci_asymmetry_score"`), NOT a human-readable title. The IO
  backfill derives the atom FQDN from `node["name"]`.
- Parameter names must match the real function parameters exactly
- Use concrete `type_desc`: `NDArray[np.float64]`, `str`, `int`, `bool`,
  `float`, `list[float] | None`
- Required params: `"required": true`. Params with defaults: `"required": false`
- Every node gets one output named `result` with the correct return type
- Use `detect_cycles` to verify acyclicity

### Phase 6: Verify

Run verification tools iteratively until clean. **Do not skip this phase.**

1. **Run `run_contribution_check(repo_root)`** — filter output for your atom
   paths. Fix all errors. Common issues: missing `@icontract.ensure`, missing
   docstrings, `Any` in public interfaces, `references.json` missing `ref_id`.
2. **Run `run_dejargon_check(repo_root)`** — fix jargon-density failures.
3. **Re-run both** after fixes until zero errors for your atoms.
4. **Functional test** — import and call each atom with realistic data. Verify
   outputs are sensible for the domain.

### Phase 7: Package Structure

Create `__init__.py` files at every new directory level:

```
src/sciona/atoms/<domain>/__init__.py
src/sciona/atoms/<domain>/<family>/__init__.py
```

### Phase 8: Add Provenance

Provenance is a two-part system: a global registry and per-atom reference files.

**Step 1: Global registry.** Add entries to `data/references/registry.json`
with short `ref_id` keys (author+year style), full citation metadata, and DOIs
when available. Also add a `repo_*` entry for the upstream source repository.

**Step 2: Per-atom references.** Write `references.json` in the family
directory using schema version 1.1. Atom keys are fully-qualified:
`<import_path>@<file_path>:<line_number>`. Each reference uses a `ref_id`
pointing to the global registry with `match_metadata`.

Do not fabricate DOIs or citations.

## Publishing

After verification and provenance, the atoms are structurally correct but not
yet publishable. **Follow [PUBLISHING.md](PUBLISHING.md)** to complete the
five publishability pillars (IO specs, parameters, descriptions, audit rollups,
references). This should be done in the same session.

## Quality Bar

### Acceptable atom (all must be true)

- [ ] Typed, registered with `@register_atom`, honestly contracted
- [ ] Every `@register_atom` atom has both `@icontract.require` and `@icontract.ensure`
- [ ] Witnesses use valid abstract types, are pure, mirror the atom surface
- [ ] CDG has concrete `inputs`/`outputs` on every atomic node
- [ ] CDG node `name` fields are snake_case function names (not human-readable)
- [ ] `__init__.py` files exist at every new directory level
- [ ] Docstrings describe real semantics, not boilerplate
- [ ] No `Any` in public interface without explicit justification
- [ ] `run_contribution_check` passes with zero errors for the new atoms
- [ ] `run_dejargon_check` passes with zero errors for the new atoms
- [ ] `references.json` uses schema v1.1 with `ref_id` keys
- [ ] Global registry entries added for all cited papers and upstream repos
- [ ] Publishing checklist in PUBLISHING.md satisfied

### Poisonous atom (reject immediately)

- Uses `Any` everywhere to suppress type errors
- Has placeholder docstrings ("This function does X" / "Input data")
- Witnesses return `None` or hardcoded values
- CDG has phantom nodes not backed by real atoms
- Contracts are tautological (`require(lambda x: x is not None)` on non-optional)
- Uses `importlib` to dynamically load the source file at runtime
- Wraps an opaque function call with no decomposition or understanding

## When to Ask the Human

Stop and ask before proceeding when:

- **Ambiguous decomposition** — multiple valid ways to split into atoms
- **Naming decisions** — domain-specific concepts where the right name matters
- **Licensing concerns** — unclear, restrictive, or missing license
- **Verification failures with no obvious fix** — deterministic fixers return
  None and you are unsure how to resolve
- **Source code has bugs** — inconsistencies that affect decomposition
- **Multiple valid decompositions** — no clear winner
- **Cross-repo dependency decisions** — which sibling repo should own the atom

## Anti-Patterns

**Opaque wrappers**: Wrapping a function call inside another function call
without understanding or decomposing the logic.

**importlib hacks**: Dynamic imports to paper over missing modules.

**Any-typed everything**: Annotating everything as `Any` to suppress mypy.

**Placeholder docstrings**: "This atom wraps the X function" is not acceptable.

**Fabricated provenance**: Do not invent DOIs or references.

**Tautological contracts**: `require(lambda x: x is not None)` on every
non-optional parameter adds noise, not safety.

**CDG fiction**: Nodes for "orchestration" that don't correspond to real atoms.

## Reference Example

See `src/sciona/atoms/causal_inference/feature_primitives/` for a complete
working example: 8 atoms with contracts, witnesses, CDG with full IO specs,
references with registry entries, review bundle, and focused tests. All
verified and publishable.

## Relationship to Other Contracts

- **INGESTION.md**: The `sciona ingest` CLI pipeline. Still valid for
  `--procedural` extraction of pure functions.
- **PUBLISHING.md**: Post-ingest steps for the five publishability pillars.
- **CONTRIBUTION.md**: The quality bar for atom PRs.
- **AGENTS.md**: Environment setup for agents.
