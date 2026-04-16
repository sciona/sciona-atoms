# REMEDIATION

This file tracks heavier-lift catalog debt that should not be papered over with relaxed publishability rules. These items need real semantic repair, better tests, or clearer scope before they should be promoted into the public catalog.

## Signal Processing

### `biosppy.svm_proc`

Status: keep unpublished for now.

Why it is blocked:
- The family is not a uniform set of thin upstream wrappers.
- Several callables are simplified local reimplementations or container shims rather than faithful behavior-preserving adapters over `biosppy.biometrics`.
- A higher publishability bar exposed real semantic drift when compared directly against upstream behavior.

What we verified:
- `cross_validation` matches upstream split behavior for deterministic test cases.
- `majority_rule` behaves plausibly for simple deterministic cases.
- `get_auth_rates` is only a partial wrapper. It returns a subset of the upstream metrics and drops important outputs such as `TAR`, `TRR`, `EER`, `Err`, `PPV`, `FDR`, `NPV`, `FOR`, and `MCC`.
- `get_id_rates` is semantically incorrect relative to upstream. The local wrapper normalizes by `H + M + R`, while upstream normalizes by total test count `N`.
- `combination` fails on ordinary upstream-shaped inputs. A normal `dict[list]` input raised `TypeError: unhashable type: 'list'` in local comparison.
- `get_subject_results`, `assess_classification`, and `assess_runs` are skeletal placeholders or lossy summaries rather than faithful behavior-preserving wrappers.

Proposed fixes:
1. Split the family into two groups.
   - Keep genuinely faithful thin wrappers as publishability candidates.
   - Mark placeholder / summarized / semantically drifted helpers as remediation-only until repaired.
2. Replace placeholder implementations with faithful upstream-aligned adapters where possible.
3. Add behavior-level tests for every retained callable by comparing concrete outputs against `biosppy.biometrics` for representative inputs.
4. Only merge review approval for rows that pass the behavior-comparison suite.
5. Consider deprecating any wrappers whose current surface cannot be made faithful without breaking callers.

Suggested remediation order:
1. `get_id_rates`
2. `combination`
3. `assess_classification`
4. `assess_runs`
5. `get_subject_results`
6. Reevaluate whether `get_auth_rates`, `majority_rule`, and `cross_validation` can be approved as a smaller faithful subset.

Evidence as of 2026-04-16:
- Direct runtime comparisons were run locally against `biosppy.biometrics` using the matcher venv.
- The comparison established semantic mismatch for `get_id_rates` and a runtime failure for `combination` on ordinary list-valued classifier results.

## Robotics

### `pronto.torque_adjustment`

Status: keep unpublished for now.

Why it is blocked:
- The family currently contains a single zero-input identity stage, `torqueadjustmentidentitystage`.
- The callable performs no observable computation, state access, or side effects and returns `None`.
- There is no meaningful behavior surface to validate beyond "does nothing," which is not a strong enough basis for publication as a reusable atom.

What we verified:
- The implementation is explicitly a no-op placeholder.
- The current witness/probe path only exercises importability, not useful robotics semantics.
- The atom is missing the full publishability bundle in Supabase because it has not been review-ratcheted and does not justify promotion as a real primitive in its current form.

Proposed fixes:
1. Decide whether this stage should exist as a first-class atom at all.
2. If it is only a graph placeholder, remove it from the publishable catalog and keep it as internal orchestration metadata instead.
3. If a real torque-adjustment primitive is intended, replace the identity stage with a semantically meaningful implementation and add behavior-level tests that exercise observable torque-adjustment logic.
4. Only reenter the publishability queue after the family has concrete inputs, outputs, references, uncertainty, and review evidence tied to real behavior.

Suggested remediation order:
1. Product/design decision on whether the stage should remain a public atom candidate.
2. Either delete/deprecate the placeholder or implement the real primitive.
3. Add behavior tests and then revisit audit review.

Evidence as of 2026-04-16:
- Local source inspection showed `torqueadjustmentidentitystage()` returns `None` and documents itself as an identity stage with no observable computation.
- No higher-signal behavior tests or meaningful metadata artifacts were present for publication review.
