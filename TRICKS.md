# Competition Tricks Catalog Implementation Plan

## Goal

Create a separate catalog for competition-specific tactics, leakage patterns, metric hacks, and other high-risk tricks so the architect can consult them only when a base CDG plus allowed expansion/refinement rounds still leaves a solution novel or under-covered.

The catalog must not pollute base CDG retrieval, deterministic CDG encoders, or expansion/refinement matching. Tricks are optional tactical context, not canonical topology.

## Phase 1: Registry Foundation

1. Add `data/solution_tricks/` in `sciona-atoms`.
2. Define `data/solution_tricks/schema.json` with a compact, stdlib-validatable object shape.
3. Add `data/solution_tricks/registry.json` with a small seed set of conservative examples.
4. Add `data/solution_tricks/GOVERNANCE.md` describing admission, risk, and graduation rules.
5. Add `scripts/validate_solution_tricks.py`.
6. Add focused tests for valid tricks, invalid enum values, missing high-risk validation requirements, duplicate IDs, CDG ID collisions, and unresolved `related_cdgs`.
7. Ensure existing CDG validation and Kaggle validation behavior is unchanged merely because tricks exist.

## Phase 2: Matcher-Side Loader

1. Add a lightweight loader in `sciona-matcher`, preferably as `sciona/principal/trick_retrieval.py` or beside `expansion_retrieval.py`.
2. Load registry data from provider roots without mixing tricks into CDG template catalogs.
3. Expose an API like:

   ```python
   retrieve_tricks(goal, candidate_cdgs, novelty_assessment, max_results=5)
   ```

4. Gate retrieval so it runs only when:
   - the best base CDG remains divergent;
   - expansion/refinement projection remains below threshold after allowed rounds;
   - or the architect explicitly marks the case as requiring a novel CDG.

## Phase 3: Architect Prompt Gating

1. Add a distinct prompt section:

   `Optional high-risk tactics for novel-CDG cases`

2. Render each trick with:
   - name
   - `kind`
   - `risk_level`
   - `generalization_level`
   - `applies_when`
   - `do_not_use_when`
   - `validation_requirements`
   - `architect_hint`
   - related CDGs/operations

3. Add prompt instructions that tricks must not be used as the base topology and must not override a better-fitting CDG or expansion/refinement path.

## Phase 4: Governance And Graduation

1. Keep high-risk tricks visible but explicitly gated.
2. Require validation requirements for `risk_level: high` and `risk_level: disallowed`.
3. Require source references or an explicit `manual_hypothesis` audit status.
4. Allow a trick to graduate into an expansion/refinement only after it recurs across unrelated competitions and changes topology or repeatable operation selection, not just leaderboard tuning.
5. Keep public leaderboard probing, target leaks, and competition artifact shortcuts cataloged for awareness, but not recommended unless the target setting explicitly permits them.

## Phase 5: Reporting

Extend validation and gap-analysis reports with trick telemetry:

- `novel_cdg_required`
- `candidate_tricks_available`
- `high_risk_tricks_suppressed`
- `tricks_consulted_by_architect`
- `tricks_used_in_plan`

This makes trick exposure auditable and helps detect overuse.

## Initial Schema

Each trick object should include:

```json
{
  "trick_id": "trick.kaggle.public_lb_probe_thresholding",
  "name": "Public leaderboard probe thresholding",
  "kind": "public_lb_overfit_risk",
  "status": "cataloged",
  "risk_level": "high",
  "generalization_level": "competition_specific",
  "summary": "Short description of the tactic.",
  "applies_when": ["Conditions where the tactic may be relevant."],
  "do_not_use_when": ["Conditions where the tactic should be suppressed."],
  "validation_requirements": ["Evidence required before using the tactic."],
  "architect_hint": "How to present this to the architect.",
  "related_cdgs": ["solution.kaggle.classical_tabular_ensemble_topology"],
  "related_operations": [],
  "source_competitions": ["example-competition-id"],
  "source_references": ["https://example.com/source"],
  "tags": ["leaderboard", "postprocessing"],
  "audit": {
    "source_kind": "manual_analysis",
    "review_status": "draft",
    "notes": "Why this is cataloged as a trick rather than a CDG."
  }
}
```

Recommended enum values:

- `kind`: `leak`, `metric_hack`, `postprocess`, `data_artifact`, `public_lb_overfit_risk`, `domain_prior`, `solver_shortcut`, `inference_budget_trick`
- `risk_level`: `low`, `medium`, `high`, `disallowed`
- `generalization_level`: `general`, `domain_specific`, `competition_specific`
- `status`: `cataloged`, `allowed_with_validation`, `deprecated`, `disallowed`

## Acceptance Criteria

Phase 1 is complete when:

- `python scripts/validate_solution_tricks.py --strict` passes.
- The new validator has focused tests.
- Existing `python scripts/validate_solution_cdgs.py --strict --errors-only` remains clean.
- No CDG matching scores change simply because the tricks registry exists.
- Seed tricks are clearly marked as optional tactical context with risk levels and validation requirements.

