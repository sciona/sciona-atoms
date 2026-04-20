# Unpublished Atom Audit Status

Generated from the live local Supabase replay on 2026-04-20T14:32:28.757169+00:00.

This document is a working debt register for every currently unpublished atom.

## Summary

- Total atoms in local catalog: `660`
- Publishable atoms: `652`
- Total non-publishable atoms in local catalog: `8`
- Remediation-excluded non-publishable atoms: `5`
- Non-publishable atoms remaining in publishability backlog: `3`


### Remediation Exclusions

- Source: `/Users/conrad/personal/sciona-atoms/REMEDIATION.md`
- `mcmc_foundational.kthohr_mcmc`: excluded `2` unpublished atoms via `prefix` match
- `scipy.sparse_graph`: excluded `3` unpublished atoms via `prefix` match

### Marginal Blocker Counts

- `description`: `3`
- `io_specs`: `2`
- `parameters`: `3`
- `publishable_rollup`: `3`
- `references`: `2`

### Top Exact Blocker Combinations

- `publishable_rollup,io_specs,parameters,description,references`: `2`
- `publishable_rollup,parameters,description`: `1`

### Largest Non-Publishable Domains

- `medical_imaging_3d`: `2`
- `robotics`: `1`

## Status Legend

- `publishable_rollup`: no approved audit rollup satisfying the current publication rule
- `io_specs`: no atom IO spec rows
- `parameters`: no atom parameter rows
- `description`: no English low-jargon description
- `references`: no atom references rows
- `missing_row`: there is no audit rollup row for the atom yet

## medical_imaging_3d

- Non-publishable atoms: `2`
- Missing publishable rollup: `2`
- Missing IO specs: `2`
- Missing parameters: `2`
- Missing description: `2`
- Missing references: `2`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.medical_imaging_3d.aggregation.casenet` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.medical_imaging_3d.aggregation.debug_atoms.casenet` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |

## robotics

- Non-publishable atoms: `1`
- Missing publishable rollup: `1`
- Missing IO specs: `0`
- Missing parameters: `1`
- Missing description: `1`
- Missing references: `0`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.robotics.pronto.torque_adjustment.apply_torque_adjustment` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |
