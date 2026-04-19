# Atom Renaming Plan

This document tracks atom renames in `sciona-atoms`, with the exact files and replay steps that must change together.

Current assumption:
- The catalog is still local-development-only.
- We can do direct renames of canonical atom names without maintaining long-lived public compatibility aliases.
- Even so, renames still need to be coordinated across source, probes, review bundles, manifest rows, and local replay artifacts.

## Scope

Completed rename wave:
- `sciona.atoms.scipy.interpolate.cubicsplinefit` -> `sciona.atoms.scipy.interpolate.cubic_spline_fit`
- `sciona.atoms.scipy.interpolate.rbfinterpolatorfit` -> `sciona.atoms.scipy.interpolate.rbf_interpolator_fit`

Why these are in scope:
- They are already published locally.
- They are the clearest examples of compressed multi-word names that should be canonical snake_case.
- They have no downstream matcher code references today, so the rename surface is contained.

Out of scope for this first wave:
- Short established upstream-aligned names such as `det`, `inv`, `butter`, `firwin`, `freqz`, `linprog`, `root`.
- Those may still be naming debt in a broad sense, but they are not compressed multi-word forms and do not justify immediate churn.

## Canonical Mapping

| Current name | Planned canonical name | Status |
| --- | --- | --- |
| `sciona.atoms.scipy.interpolate.cubicsplinefit` | `sciona.atoms.scipy.interpolate.cubic_spline_fit` | implemented 2026-04-16 |
| `sciona.atoms.scipy.interpolate.rbfinterpolatorfit` | `sciona.atoms.scipy.interpolate.rbf_interpolator_fit` | implemented 2026-04-16 |

## Shared Repo Touch Points

These files were updated together for the interpolate rename wave completed on 2026-04-16.

### 1. Source wrappers

File:
- `src/sciona/atoms/scipy/interpolate.py`

Required changes:
- Rename `witness_cubicsplinefit` -> `witness_cubic_spline_fit`
- Rename `witness_rbfinterpolatorfit` -> `witness_rbf_interpolator_fit`
- Rename `cubicsplinefit` -> `cubic_spline_fit`
- Rename `rbfinterpolatorfit` -> `rbf_interpolator_fit`
- Update `register_atom(...)` bindings accordingly
- Update `__all__`

Notes:
- Because the interpolate wrappers currently do not pass explicit `name=...`, the exported atom FQDN is driven by the Python function name. Renaming the function is the canonical rename.

### 2. Probe catalog

File:
- `src/sciona/probes/scipy/interpolate.py`

Required changes:
- Update `atom_fqdn` values to the new FQDNs
- Update `wrapper_symbol` values to `cubic_spline_fit` and `rbf_interpolator_fit`

### 3. Import smoke and probe tests

Files:
- `tests/test_scipy_interpolate_import_smoke.py`
- `tests/test_scipy_interpolate_probes.py`

Required changes:
- Update `hasattr(...)` assertions
- Update expected wrapper symbol sets
- Update any explicit FQDN expectations

### 4. Review-bundle coverage test

File:
- `tests/test_review_bundles.py`

Required changes:
- Update the expected atom keys for `scipy_interpolate.review_bundle.json`

### 5. Review bundle

File:
- `data/review_bundles/scipy_interpolate.review_bundle.json`

Required changes:
- Update each row `atom_key`
- Keep the same evidence paths unless source/test file names also change

Notes:
- If the file names stay the same and only wrapper symbols/FQDNs change, the evidence paths should remain stable.

### 6. Audit manifest

File:
- `data/audit_manifest.json`

Required changes for the canonical rows:
- Update `atom_name`
- Update `atom_key`
- Update `atom_id`
- Update `wrapper_symbol`
- Update `witness_binding`
- Update any string fields that embed the old names

Current canonical rows that must be renamed:
- `sciona.atoms.scipy.interpolate.cubicsplinefit`
- `sciona.atoms.scipy.interpolate.rbfinterpolatorfit`

Additional manifest cleanup tied to this rename:
- Remove stale legacy rows under:
  - `sciona.atoms.scipy.interpolate_v2.cubicsplinefit`
  - `sciona.atoms.scipy.interpolate_v2.rbfinterpolatorfit`

Why this matters:
- The manifest should carry only the live snake_case canonical interpolate rows.
- The stale `interpolate_v2` rows were temporarily useful for carrying upstream/reference metadata into the new canonical rows, but they are no longer the source of truth after the rename wave landed.

### 7. Local documentation and debt tracking

Files:
- `REMEDIATION.md`
- `docs/RENAMING_ATOMS.md`

Required changes:
- After the rename lands, update any prose that cites the old interpolate names as examples of naming debt.

## Publishability Audit Touch Points

Current state:
- No matcher code references were found for `cubicsplinefit` or `rbfinterpolatorfit`.
- The impact in `sciona-matcher` is therefore operational, not code-level.

Files affected by refresh:
- `docs/audit/UNPUBLISHED_ATOM_AUDIT_STATUS.md`
- `docs/audit/unpublished_atom_audit_status.json`
- `docs/audit/PUBLISHABILITY_REVIEW_BATCH_QUEUE.md`
- `docs/audit/publishability_review_batch_queue.json`

Required action:
- Refresh publishability docs after replay so the renamed FQDNs are reflected anywhere the backlog or publishability state surfaces them.

## Replay / Validation Steps

After a rename is implemented in shared-atoms:

1. Run focused tests in `sciona-atoms`
   - `python -m pytest tests/test_review_bundles.py tests/test_scipy_interpolate_import_smoke.py tests/test_scipy_interpolate_probes.py -q`
2. Merge review bundles into the shared manifest
   - `scripts/apply_audit_review_bundles.py`
3. Reset local Supabase
   - `supabase db reset --local --yes`
4. Seed from shared-atoms
   - `scripts/supabase_seed.py --apply --ensure-owner`
5. Backfill
   - `scripts/supabase_backfill.py all-file-backed`
6. Refresh publishability docs
   - `scripts/refresh_publishability_review_docs.py`
7. Verify in DB
   - check the new interpolate FQDNs are present and publishable
   - confirm the old interpolate FQDNs are gone or intentionally absent

## Follow-On Legacy Cleanup

These are not part of the immediate interpolate rename wave, but they should be cleaned up with the same discipline because they still encode compressed names in the repo.

### Spatial legacy aliases / stale rows

Current legacy names:
- `delaunaytriangulation`
- `voronoitessellation`

Current locations:
- `src/sciona/atoms/scipy/spatial.py`
  - compatibility aliases only
- `data/audit_manifest.json`
  - stale `scipy.spatial_v2.*` rows

Target:
- keep `delaunay_triangulation` and `voronoi_tessellation` as the only canonical names

### Optimize legacy aliases / stale rows

Current legacy names:
- `differentialevolutionoptimization`
- `shgoglobaloptimization`

Current locations:
- `src/sciona/atoms/scipy/optimize.py`
  - compatibility aliases
- `src/sciona/atoms/scipy/witnesses.py`
  - witness helper names
- `data/audit_manifest.json`
  - stale `scipy.optimize_v2.*` rows

Target:
- keep `differential_evolution` / `differential_evolution_optimization` and `shgo` / `shgo_global_optimization` as the only canonical spellings, depending on final publication policy for that family

## Recommended Order

1. Replay and refresh until the new interpolate names are publishable.
2. Confirm the old interpolate names are absent from both the manifest and the local catalog.
3. Clean up legacy spatial and optimize alias debt in a separate wave.

## Non-Goals

This document does not recommend renaming every short SciPy atom.

Examples that should remain unchanged unless a separate policy says otherwise:
- `det`
- `inv`
- `dct`
- `idct`
- `butter`
- `firwin`
- `freqz`
- `norm`

Reason:
- These are established upstream-facing names, not compressed multi-word names.
