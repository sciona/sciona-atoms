# Solution Tricks Governance

The tricks registry catalogs competition-specific tactics that may be useful context for an architect LLM after normal CDG matching, expansion, and refinement have failed to cover a solution.

Tricks are not CDGs. They must not participate in base CDG ranking, deterministic CDG encoding, or expansion/refinement selection unless a separate architect step explicitly asks for optional tactical context.

## Admission Rules

- Add a trick only when it explains a recurring tactic, benchmark artifact, leakage risk, or metric-specific maneuver that should remain outside canonical CDG topology.
- Do not add a trick solely because it improves one validation case.
- Do not use tricks to duplicate base CDG structure or first-class expansion/refinement operations.
- Use `risk_level: high` for public leaderboard probing, hidden target leakage, private-label inference, or tactics that can invalidate validation.
- Use `status: disallowed` when a tactic should be visible for detection/suppression but not recommended.
- Include `validation_requirements` for every trick; high-risk and disallowed tricks require concrete validation or suppression requirements.

## Architect Exposure

Expose tricks only when:

- the best base CDG is still divergent;
- allowed expansion/refinement rounds remain below coverage threshold;
- or the architect explicitly marks the case as requiring a novel CDG.

Render tricks in a separate section named `Optional high-risk tactics for novel-CDG cases`.

The architect prompt must state that tricks are tactical context and must not determine the base topology.

## Graduation Rules

A trick can graduate into an expansion/refinement operation only when:

- it recurs across unrelated competitions;
- it has a stable input/output contract;
- it changes a repeatable operation or refinement path rather than only tuning a leaderboard;
- and it passes semantic comparison against the closest existing CDGs and operations.

A trick can graduate into a new base CDG only when it represents a distinct reusable topology, not a shortcut, leakage pattern, or metric-specific post-processing rule.

Graduation review is recorded with an optional `graduation` block. Omit the block when no review has happened yet. When present, use:

```json
{
  "candidate_kind": "none",
  "decision": "not_reviewed",
  "recurrence_evidence": [],
  "stable_io_contract": false,
  "repeatable_operation_change": "",
  "distinct_reusable_topology": false,
  "semantic_comparison": [],
  "target_asset_id": ""
}
```

For `candidate_kind: expansion_operation`, the validator requires:

- `recurrence_evidence` from at least two unrelated competitions;
- `stable_io_contract: true`;
- a non-empty `repeatable_operation_change`;
- non-empty `semantic_comparison` against the closest CDGs or operations;
- `risk_level` below `high`.

For `candidate_kind: base_cdg`, the validator also requires `distinct_reusable_topology: true`.

Leakage and public-leaderboard probing tricks cannot graduate into canonical CDG or expansion assets. Keep them cataloged for detection/suppression only.

## Review Checklist

- The trick has a unique `trick_id`.
- The trick has clear `applies_when` and `do_not_use_when` boundaries.
- The trick names its validation requirements.
- The related CDGs exist.
- High-risk tricks are clearly labeled and do not include instructions to exploit prohibited data.
- Source references or local audit references are present.
- Graduation candidates include recurrence, semantic comparison, and stable contract evidence.
