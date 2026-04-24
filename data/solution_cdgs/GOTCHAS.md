# Solution CDG Ingestion — Gotcha Log

This file tracks every design decision, unexpected issue, and architect
tooling gap discovered while converting Kaggle winning solutions into CDGs.

## Format Decisions

### G-001: Paradigm field for multi-paradigm solutions
**Date:** 2026-04-22
**Context:** SkeletonFamilyAsset requires a `paradigm` field from ConceptType enum.
Kaggle solutions span multiple paradigms (e.g., signal_filter + dimensionality_reduction + analysis).
**Decision:** Use the DOMINANT paradigm for the overall solution. Individual stages
get their own concept_type. For solutions that are truly multi-paradigm pipelines,
use `analysis` as the top-level paradigm since it's the most general.
**Risk:** The architect's strategy selector uses paradigm to narrow candidates.
A solution CDG tagged `analysis` may not be retrieved when the architect is
looking for `signal_filter` templates. Need to verify whether `variant_hints`
can compensate.

### G-002: Solution CDGs use SkeletonFamilyAsset format
**Date:** 2026-04-22
**Context:** Could have invented a custom format, but `skeleton_catalog_sync.py`
is the proven import path to Supabase. The `SkeletonFamilyAsset` schema maps
directly to `artifact_cdg_nodes`, `artifact_cdg_edges`, and `artifact_cdg_bindings`.
**Decision:** Use the skeleton asset JSON format exactly. Solution CDGs are
skeleton assets where `matched_primitive` references existing atom FQDNs (or is
empty for gaps).
**Risk:** Skeleton assets are designed as TEMPLATES (abstract stages filled by
the architect), but solution CDGs are CONCRETE (specific atoms already chosen).
The `matched_primitive` field handles this — when populated, the architect
treats it as a pre-bound stage.

### G-003: Binding confidence not in skeleton format
**Date:** 2026-04-22
**Context:** The DB has `artifact_cdg_bindings.binding_confidence` (float) and
`binding_source` (text), but the skeleton JSON format only has `matched_primitive`
(string, the FQDN). There's no way to express "this is an approximate match" or
"this is a gap" in the JSON alone.
**Decision:** Use a companion `_bindings.json` file per solution CDG that maps
stage_id → {bound_artifact_fqdn, binding_confidence, binding_source, status,
action_class, evidence}. The skeleton JSON carries the FQDN in matched_primitive
for convenience; the bindings file carries the full metadata for DB hydration.
**Risk:** Two files to keep in sync. Acceptable for 7 solutions.

### G-004: Edge format — skeleton edges have richer semantics than CDG edges
**Date:** 2026-04-22
**Context:** SkeletonEdgeAsset has `data_kind`, `provenance`, `loss_class`,
`alignment_expectation` — much richer than the DB's `artifact_cdg_edges` which
just stores source_id/target_id/output_name/input_name. The richer fields help
the architect reason about data flow.
**Decision:** Populate the rich edge fields where applicable. They're used by
the architect's planner even if not stored in the DB.

### G-005: SkeletonGraph runtime attribute naming
**Date:** 2026-04-22
**Context:** `SkeletonFamilyAsset.to_skeleton_graph()` returns a `SkeletonGraph`
whose attribute for nodes is not `.nodes` (likely `.stage_nodes` or similar).
**Impact:** Low — the JSON parsing and DB hydration path via `skeleton_catalog_sync`
works on the asset, not the runtime graph. Only matters if we try to use
solution CDGs in the architect's live decomposition loop.
**Action needed:** Check `SkeletonGraph` model for correct attribute name before
using solution CDGs in live architect tests.

## Atom Binding Issues — Connectomics 1st

### B-001: PCA precision matrix — partial coverage
**Date:** 2026-04-22
**Context:** `pca_precision_matrix` stage binds to `pca_fit` (confidence 0.7), but
the critical step is `get_precision()` on the fitted PCA model, which is NOT
covered by any existing atom. The atom does PCA decomposition; extracting the
precision matrix is a separate state query.
**Gap type:** Missing state-query atom (`pca_get_precision`).
**Architect test prediction:** The architect SHOULD find `pca_fit` via
concept_type=dimensionality_reduction search, but will not find the precision
extraction step. This will surface as an incomplete binding.

### B-002: VarianceThreshold false positive risk
**Date:** 2026-04-22
**Context:** Keyword search for "threshold" returns `variance_threshold_fit`, but
this is a COLUMN-WISE feature selector, not an ELEMENT-WISE value clipper. The
semantic mismatch could mislead the architect.
**Gap type:** Misleading keyword overlap between semantically different operations.
**Architect test prediction:** If the architect searches by keyword "threshold" +
concept_type "signal_filter", it should NOT return VarianceThreshold. If it does,
that's a retrieval quality bug.

### B-003: threshold_sweep_ensemble is an orchestration pattern
**Date:** 2026-04-22
**Context:** This stage wraps the inner pipeline in a loop over 120×4 parameter
combinations. It's not a single atom — it's a MAP_OVER or BRANCH_AND_COMPARE
meta-pattern. The current atom catalog has no higher-order composition primitives.
**Gap type:** Missing orchestration/composition primitives in the atom catalog.
**Architect test prediction:** The architect should recognize this as a MAP_OVER
concept_type and not attempt to bind it to a single atom.

(More binding issues to be added per solution)
