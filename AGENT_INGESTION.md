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

Import all tools from `sciona.tools`:

```python
from sciona.tools import (
    extract_dfg, extract_function_dfg, extract_procedural_dfg,
    run_mypy, classify_type_failure, build_type_fixes,
    classify_ghost_failure, build_ghost_fixes,
    detect_cycles, break_cycle,
    validate_cdg_ir,
    match_witness_template, generate_abstract_profile,
    run_contribution_check, run_dejargon_check,
)
```

### Extraction Tools

**`extract_dfg(source_path, class_name) -> RawDataFlowGraph`**

Start here for class-based ingestion. Parses the source file with Python's AST
module and builds deterministic data-flow facts: method signatures, attribute
access patterns, cross-window state, fitted/config classification, and the
internal call graph.

Read the result to understand what the class does before deciding how to
decompose it. Key fields:
- `methods`: list of `MethodFact` — one per method, with params, reads, writes,
  calls, return facts, and semantic role
- `all_attributes`: attribute access patterns
- `fitted_attributes`: attributes written by fit/update, read by predict/transform
- `config_attributes`: attributes set in `__init__` and never mutated
- `cross_window_attrs`: state that persists across calls

**`extract_function_dfg(source_path, function_name) -> RawDataFlowGraph`**

Same as above but targets a module-level function. Use for standalone function
ingestion.

**`extract_procedural_dfg(source_path) -> RawDataFlowGraph`**

Deterministic SSA-based extraction for scripts and simple functions. No LLM
calls. Best for pure functions with clear data flow.

### Type Verification Tools

**`run_mypy(source_files, *, strict=True) -> str`**

Run after writing `atoms.py` and `witnesses.py`. Pass a dict of filename →
source content. Returns the raw mypy error output. Empty string means clean.

```python
errors = run_mypy({
    "atoms.py": atoms_source,
    "witnesses.py": witness_source,
})
```

**`classify_type_failure(mypy_errors, source_files) -> dict`**

If mypy fails, classify the errors before attempting fixes. The result tells
you whether errors are mechanically repairable (missing imports, annotation
issues) or semantic (signature mismatches, output binding errors).

Key field: `result["repairable"]` — if True, try `build_type_fixes`.

**`build_type_fixes(mypy_errors, source_files) -> list[dict] | None`**

Attempt deterministic patches for mypy errors. Returns a list of
`{file, line_start, line_end, replacement}` patches, or None if no
deterministic fix applies. If None, fix the errors manually.

### Ghost Verification Tools

**`classify_ghost_failure(report, witness_source) -> dict`**

Classify a ghost simulation failure. Same structure as type failure classifier.

**`build_ghost_fixes(error_node, error_function, error_message, witness_source) -> list[dict] | None`**

Attempt deterministic fixes for ghost simulation failures. Handles None-return,
TypeError, KeyError, and AttributeError patterns.

### Cycle Detection

**`detect_cycles(node_ids, edges) -> set[str]`**

Check a CDG for cycles before running ghost simulation. Pass the set of node
IDs and list of `(source_id, target_id)` edge tuples. Returns the set of nodes
participating in cycles, or an empty set if acyclic.

**`break_cycle(deadlock_nodes, cycle_edges, witness_source) -> tuple | None`**

If cycles exist in witnesses, try deterministic patching. Returns
`(patches, strategy_name)` or None.

### CDG Validation

**`validate_cdg_ir(dfg, ir) -> tuple[bool, str, list[str]]`**

Validate that a canonical IR plan covers all attributes, has consistent state
slots, and contains no forbidden cycles. Use after writing `cdg.json`.

### Template Matching

**`match_witness_template(class_name, base_classes, method_name, params, return_type, docstring) -> tuple[str, str] | None`**

For opaque DL layers (PyTorch, JAX, TF), check if a witness template matches
before writing a custom witness. Returns `(shape_transform, witness_body)` or
None.

**`generate_abstract_profile(atom_name, concept_type, inputs, outputs, methods) -> dict | None`**

Generate a cross-domain conceptual profile. Useful for the `conceptual_summary`
field in CDG nodes.

### Audit Tools

**`run_contribution_check(repo_root) -> tuple[bool, str]`**

Run the contribution rules validator on the target repo. Must pass before
declaring the atom ready.

**`run_dejargon_check(repo_root) -> tuple[bool, str]`**

Run the dejargon validator. Checks that docstrings are readable outside the
source discipline.

## Workflow Phases

These are recommended phases, not a rigid pipeline. Use judgment about which
phases to skip or reorder based on the source code.

### Phase 1: Read and Understand

Use `extract_dfg` or `extract_function_dfg` to understand the source. Read the
DFG output carefully:

- What methods exist? What are their semantic roles?
- What state is fitted vs config vs transient?
- What does the internal call graph look like?
- Are there unknowns or ambiguities in the extraction?

Also read the source code directly. The DFG is a structural summary — the
code reveals intent, naming conventions, and domain semantics that the AST
cannot capture.

### Phase 2: Decide on Decomposition

- **Which atoms?** Each method with a clear, separable role typically becomes
  one atom. Helper methods called by only one public method should be inlined,
  not separate atoms.
- **Which family?** Group atoms that share state, imports, or conceptual scope
  into one family directory with a shared `atoms.py` and `witnesses.py`.
- **State model needed?** If there is cross-window state (attributes written in
  one method, read in another), you need `state_models.py`.
- **Uncertain?** Ask the human about decomposition boundaries.

### Phase 3: Write atoms.py

Follow the pattern in existing atoms. Every public atom must:

- Use `@register_atom(witness_fn)` as the outermost decorator
- Include `@icontract.require` preconditions (meaningful, not tautological)
- Include `@icontract.ensure` postconditions
- Be fully type-annotated — no `Any` in public interfaces unless justified
- Have an honest docstring describing what it computes, not boilerplate

Reference existing atoms in the repo for the exact import and decorator pattern.

### Phase 4: Write witnesses.py

Witnesses are pure, side-effect-free functions that mirror the atom interface
using abstract ghost types. They describe the semantic shape of the computation
without executing it.

- Use `AbstractArray` for tensor parameters and return values
- Capture shape transforms symbolically
- Keep witnesses minimal — no real computation
- For DL layers, try `match_witness_template` first

### Phase 5: Write cdg.json

The CDG captures dependency structure between atoms in a family. Nodes
represent atoms; edges represent data and state dependencies.

- Use `detect_cycles` to verify acyclicity
- Use `validate_cdg_ir` if you have a full IR plan

For single-atom families (standalone functions), the CDG is a single root node
with one atomic child and no edges.

### Phase 6: Verify

Run verification tools iteratively until clean. **Do not skip this phase.**
Atoms that fail verification are not publishable.

1. **Run `run_contribution_check(repo_root)`** — filter the output for your
   new atom paths. Fix all errors. Common issues:
   - Missing `@icontract.ensure` on `@register_atom` functions
   - Missing docstrings on public functions
   - `Any` in public interfaces
   - `references.json` entries missing `ref_id`
2. **Run `run_dejargon_check(repo_root)`** — filter for your atom paths.
   Fix any jargon-density failures in docstrings.
3. **Re-run both checks** after fixes until your atoms produce zero errors.
4. **Functional test** — write a short script that imports and calls each atom
   with realistic data. Verify the outputs are sensible for the domain. This
   catches import errors, runtime bugs, and contract violations that static
   checks miss.

### Phase 7: Package Structure

Create `__init__.py` files at every new directory level so imports resolve:

```
src/sciona/atoms/<domain>/__init__.py
src/sciona/atoms/<domain>/<family>/__init__.py
```

These can be empty files.

### Phase 8: Add Provenance

Provenance is a two-part system: a global registry and per-atom reference files.

**Step 1: Add entries to the global references registry.**

The registry lives at `data/references/registry.json`. Each entry has a short
`ref_id` key (author+year style) and full citation metadata:

```json
{
  "ref_id": "daniusis2012igci",
  "type": "paper",
  "title": "Inferring deterministic causal relations",
  "authors": ["Povilas Daniusis", "Dominik Janzing", "..."],
  "year": 2012,
  "venue": "UAI 2012",
  "doi": "10.48550/arXiv.1203.3475",
  "match_metadata": {
    "similarity_score": null,
    "match_type": "manual",
    "matched_nodes": [],
    "confidence": "high",
    "notes": "IGCI framework for causal direction inference."
  }
}
```

Also add a `repo_*` entry for the upstream source repository when applicable.
Do not fabricate DOIs or citations. If no authoritative source exists, skip
this step.

**Step 2: Write per-atom `references.json` in the family directory.**

Use schema version 1.1. Atom keys are fully-qualified:
`<import_path>@<file_path>:<line_number>`. Each reference uses a `ref_id`
pointing to the global registry, with `match_metadata` describing the
relationship:

```json
{
  "schema_version": "1.1",
  "atoms": {
    "sciona.atoms.<domain>.<family>.<atom_name>@sciona/atoms/<domain>/<family>/atoms.py:<line>": {
      "references": [
        {
          "ref_id": "<registry_key>",
          "match_metadata": {
            "similarity_score": null,
            "match_type": "manual",
            "matched_nodes": [],
            "confidence": "high",
            "notes": "Description of how this atom relates to the reference."
          }
        }
      ],
      "auto_attribution_runs": []
    }
  }
}
```

Get the line numbers by checking where each `def <atom_name>` appears in
`atoms.py`.

## Quality Bar

### Acceptable atom (all must be true)

- [ ] Typed, registered with `@register_atom`, honestly contracted
- [ ] Every `@register_atom` atom has both `@icontract.require` and `@icontract.ensure`
- [ ] Witnesses use valid abstract types, are pure, mirror the atom surface
- [ ] CDG matches the actual decomposition
- [ ] `__init__.py` files exist at every new directory level
- [ ] Docstrings describe real semantics, not boilerplate
- [ ] No `Any` in public interface without explicit justification
- [ ] `run_contribution_check` passes with zero errors for the new atoms
- [ ] `run_dejargon_check` passes with zero errors for the new atoms
- [ ] `references.json` uses schema v1.1 with `ref_id` keys pointing to `data/references/registry.json`
- [ ] Global registry entries added for all cited papers and upstream repos

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

- **Ambiguous decomposition** — multiple valid ways to split a class into atoms,
  or methods that serve multiple roles
- **Naming decisions** — domain-specific concepts where the right name matters
  for discoverability and matching
- **Licensing concerns** — source code with unclear, restrictive, or missing
  license
- **Verification failures with no obvious fix** — deterministic fixers return
  None and you are unsure how to resolve the error
- **Source code has bugs** — inconsistencies, dead code, or undocumented behavior
  that affects decomposition
- **Multiple valid decompositions** — no clear winner; the human should decide
  based on how the atoms will be used
- **Cross-repo dependency decisions** — which sibling repo should own the atom

## Anti-Patterns

These patterns indicate degraded or poisoned output. Avoid them:

**Opaque wrappers**: Wrapping a function call inside another function call to
satisfy the atom interface without understanding or decomposing the logic.

**importlib hacks**: Using `importlib.util.spec_from_file_location` to load
source modules at runtime. Atoms must be self-contained.

**Any-typed everything**: Annotating all parameters and returns as `Any` to
suppress mypy. If you cannot determine the type, ask the human.

**Placeholder docstrings**: "This atom wraps the X function" or "Input data" are
not acceptable. Describe what the function computes, what the inputs represent,
and what the output means.

**Fabricated provenance**: Do not invent DOIs, paper titles, or references. If
no authoritative source exists, omit `references.json`.

**Shadow witness stubs**: Do not define local witness functions that bypass the
real witness system. Witnesses must be in `witnesses.py`.

**Tautological contracts**: `require(lambda x: x is not None)` on every
parameter when the type already excludes `None` adds noise, not safety.

**CDG fiction**: Do not create CDG nodes for "orchestration" or "pipeline" that
do not correspond to real atom functions.

## Worked Example

Ingesting the `igci` function from a causal inference competition solution
(Apache 2.0 licensed, `github.com/jarfo/cause-effect`).

### 1. Read the source

Read the source file directly. Understand what the function computes, what its
inputs/outputs mean, what helpers it depends on, and what the domain semantics
are. The `extract_function_dfg` tool can supplement this with structural facts:

```python
from sciona.tools import extract_function_dfg
dfg = extract_function_dfg("third_party/cause-effect-2nd/features.py", "igci")
```

From reading the source: `igci(x, tx, y, ty)` implements Information-Geometric
Causal Inference — sorts by X, computes log-ratio of consecutive deltas. It
depends on a `normalize` helper and handles non-injective mappings by averaging
Y values per unique X.

### 2. Decide on decomposition

Single pure function, no state. Goes in the `causal_inference/feature_primitives`
family alongside related causal feature functions. Shared helpers (`normalize`,
`to_numerical`, type constants) become private functions in the module.

### 3. Create directory structure

```
src/sciona/atoms/causal_inference/__init__.py          (empty)
src/sciona/atoms/causal_inference/feature_primitives/__init__.py  (empty)
```

### 4. Write atoms.py

Key requirements: `@register_atom` outermost, both `@icontract.require` and
`@icontract.ensure` present, full type annotations, honest docstring.

```python
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom
from .witnesses import witness_igci_asymmetry_score

@register_atom(witness_igci_asymmetry_score)
@icontract.require(lambda x, y: len(x) == len(y), "x and y must have equal length")
@icontract.require(lambda x: len(x) >= 2, "need at least 2 samples")
@icontract.ensure(lambda result: np.isfinite(result), "score must be finite")
def igci_asymmetry_score(
    x: NDArray[np.float64],
    tx: str,
    y: NDArray[np.float64],
    ty: str,
) -> float:
    """Compute the IGCI asymmetry score for causal direction inference.

    Sorts samples by x, computes the log-ratio of consecutive x-deltas
    to y-deltas. Under the IGCI framework (Daniusis et al., 2012), if
    X causes Y via a deterministic function, the slope distribution is
    independent of the cause distribution. Handles non-injective mappings
    by averaging y values per unique x.

    Args:
        x: Cause candidate variable, shape (n,).
        tx: Type descriptor (BINARY, CATEGORICAL, or NUMERICAL).
        y: Effect candidate variable, shape (n,).
        ty: Type descriptor for y.

    Returns:
        IGCI asymmetry score. Compute igci(X,Y) - igci(Y,X) for direction.
    """
    # ... implementation with _normalize helper ...
```

### 5. Write witnesses.py

```python
from sciona.ghost.abstract import AbstractArray

def witness_igci_asymmetry_score(
    x: AbstractArray, tx: str, y: AbstractArray, ty: str
) -> float:
    """Ghost witness for IGCI asymmetry score.

    Takes two equal-length arrays and type descriptors, returns a scalar
    float score. Shape-invariant: output does not depend on input shape.
    """
    return 0.0
```

### 6. Write cdg.json

```json
{
  "nodes": [
    {"node_id": "causal_feature_primitives_root", "name": "Causal Feature Primitives",
     "status": "decomposed", "children": ["igci_asymmetry_score"]},
    {"node_id": "igci_asymmetry_score", "name": "IGCI Asymmetry Score",
     "status": "atomic", "concept_type": "custom",
     "type_signature": "(x: NDArray, tx: str, y: NDArray, ty: str) -> float"}
  ],
  "edges": [],
  "metadata": {"source": "agent_ingestion", "source_license": "Apache-2.0"}
}
```

### 7. Verify

Run contribution rules and dejargon checks. Filter output for your atoms:

```bash
../sciona-matcher/.venv/bin/python scripts/verify_contribution_rules.py --repo-root . 2>&1 | grep causal_inference
../sciona-matcher/.venv/bin/python scripts/validate_dejargon.py --root . 2>&1 | grep causal_inference
```

Fix any errors. Common issues at this stage:
- Missing `@icontract.ensure` → add a meaningful postcondition
- `references.json` missing `ref_id` → use the proper schema (see Phase 8)

Re-run until clean: zero errors for your new atoms.

Then run a functional test:

```python
import numpy as np
from sciona.atoms.causal_inference.feature_primitives.atoms import (
    igci_asymmetry_score, NUMERICAL
)
x = np.random.randn(200)
y = x**2 + 0.1 * np.random.randn(200)
score = igci_asymmetry_score(x, NUMERICAL, y, NUMERICAL)
print(f"igci score: {score:.4f}")  # should be finite, non-zero
```

### 8. Add provenance

**Global registry** — add entry to `data/references/registry.json`:

```json
"daniusis2012igci": {
  "ref_id": "daniusis2012igci",
  "type": "paper",
  "title": "Inferring deterministic causal relations",
  "authors": ["Povilas Daniusis", "Dominik Janzing", "..."],
  "year": 2012,
  "venue": "UAI 2012",
  "doi": "10.48550/arXiv.1203.3475",
  "match_metadata": {
    "similarity_score": null, "match_type": "manual",
    "matched_nodes": [], "confidence": "high",
    "notes": "IGCI framework for causal direction inference."
  }
}
```

**Per-atom references** — write `references.json` in the family directory using
fully-qualified atom keys (`<import_path>@<file_path>:<line_number>`):

```json
{
  "schema_version": "1.1",
  "atoms": {
    "sciona.atoms.causal_inference.feature_primitives.igci_asymmetry_score@sciona/atoms/causal_inference/feature_primitives/atoms.py:139": {
      "references": [
        {
          "ref_id": "daniusis2012igci",
          "match_metadata": {
            "similarity_score": null, "match_type": "manual",
            "matched_nodes": [], "confidence": "high",
            "notes": "Direct implementation of the IGCI estimator."
          }
        },
        {
          "ref_id": "repo_cause_effect",
          "match_metadata": {
            "similarity_score": null, "match_type": "manual",
            "matched_nodes": [], "confidence": "high",
            "notes": "Upstream source: features.py:igci()"
          }
        }
      ],
      "auto_attribution_runs": []
    }
  }
}
```

### 9. Final check

Re-run `verify_contribution_rules.py` one last time to confirm the references
pass validation. The atom is now publishable.

## Relationship to Other Contracts

- **INGESTION.md**: Defines the `sciona ingest` CLI pipeline. The CLI is still
  valid for `--procedural` extraction of pure functions. This guide is for
  complex ingestion that benefits from agent judgment.
- **CONTRIBUTION.md**: Defines the quality bar. This guide references it but
  expresses the bar as agent-actionable checks.
- **AGENTS.md**: Environment setup. This guide extends it with tool-specific
  setup.
