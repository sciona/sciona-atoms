# REMEDIATION

This file tracks heavier-lift catalog debt that should not be papered over with relaxed publishability rules. These items need real semantic repair, better tests, or clearer scope before they should be promoted into the public catalog.

## Inference

### `mcmc_foundational.kthohr_mcmc` auto-generated MCMC wrapper rows

Status: keep the listed KTHOHR MCMC rows unpublished for now.

Held atoms:
- `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.aees.metropolishastingstransitionkernel`
- `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.aees.targetlogkerneloracle`
- `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.de.build_de_transition_kernel`
- `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.hmc.buildhmckernelfromlogdensityoracle`
- `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.mala.mala_proposal_adjustment`
- `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.mcmc_algos.dispatch_mcmc_algorithm`
- `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.nuts.nuts_recursive_tree_build`
- `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.rmhmc.buildrmhmctransitionkernel`
- `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.rwmh.constructrandomwalkmetropoliskernel`

Why they are blocked:
- The wrappers are explicitly marked “Auto-generated” and several expose placeholder or semantically drifted behavior relative to their public names and docstrings.
- `aees.targetlogkerneloracle` is a hard-coded toy scoring function (`temper_val * sum(state_candidate)`) and does not accept or bind to a real target distribution/log-density oracle.
- `aees.metropolishastingstransitionkernel` conflates “RNG key” with the sampler state: the only array input (`rng_key_in`) is treated as the current state vector and also used to seed randomness, so there is no explicit state-in input for an MH transition.
- `mcmc_algos.dispatch_mcmc_algorithm` treats `log_target_density` as a flat numeric array and uses a proxy log-probability update (mean density signal times proposal delta) rather than evaluating an actual log-density oracle; this is not a defensible “dispatch” primitive.
- `nuts.nuts_recursive_tree_build` returns only a position-like array while the docstring advertises a structured NUTS trajectory; it also contains explicit placeholder kinetic energy.
- The witness modules do not substantiate the callable surfaces (several witnesses model oracles as `AbstractArray`/extra args rather than `AbstractSignal`/callables), so the witness system is not currently validating the advertised contracts.
- File-backed publishability metadata is currently non-canonical for this family, compounding the semantic risk:
  - `references.json` keys omit the `inference` namespace segment (and use stale `@sciona/atoms/...` paths), so reference backfill does not bind to the catalog FQDNs.
  - The existing `data/review_bundles/mcmc_foundational.review_bundle.json` rows for `kthohr_mcmc` use non-FQDN `atom_key` values (path-like identifiers), which the manifest merge tool skips as “unresolved atoms”.
  - The per-module `*_cdg.json` files sit next to `de.py`, `hmc.py`, etc., so the IO backfill’s `derive_atom_fqdn` logic cannot infer module-qualified FQDNs like `...kthohr_mcmc.de.build_de_transition_kernel` from these CDG file paths.

Proposed fixes:
1. Decide and document intended semantics:
   - either real bindings to `kthohr/mcmc` (and ship/locate the compiled library + stable FFI), or
   - explicitly scoped educational/minimal NumPy samplers with names/docstrings that do not claim full upstream parity.
2. Repair the AEES row interfaces so state and RNG are explicit and separable (e.g., `(state_in, rng_in) -> (state_out, rng_out)`), and replace/remove `targetlogkerneloracle` unless it can bind to a real target oracle.
3. Replace `dispatch_mcmc_algorithm` with an honest API: accept a log-density oracle (callable) and select among concrete kernel builders, or rename it to reflect the actual implemented behavior.
4. Either implement a real NUTS tree builder returning a structured trajectory record (left/right states, candidate, stop/divergence flags, acceptance stats), or rename and re-document the atom to match the current “rightmost-state recursion” semantics.
5. Canonicalize metadata and make it ingestible:
   - Update `references.json` keys to the catalog FQDNs under `sciona.atoms.inference...` and ensure all referenced `ref_id`s exist in `data/references/registry.json`.
   - Update review bundle rows to use canonical `atom_name`/`atom_key` FQDNs and add focused family tests (bundle mergeability, references registry, and minimal behavior smoke).
   - Fix IO-spec generation for module-scoped CDGs (either relocate CDG files under per-module subdirectories, or update the backfill naming scheme for `*_cdg.json` in a controlled way and add regression tests).

Evidence as of 2026-04-19:
- Local wrapper inspection shows explicit placeholder logic and contract drift in the AEES, dispatch, and NUTS rows.
- The current publishability backlog marks all nine atoms as missing rollups, IO specs, parameters, descriptions, and references, consistent with the non-canonical bundle/sidecar metadata shapes described above.

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

## Bio

### `molecular_docking.quantum_solver.adiabaticquantumsampler`

Status: keep unpublished for now.

Why it is blocked:
- The current path is a classical placeholder/approximation rather than the advertised quantum or adiabatic solver behavior.

### `molecular_docking.quantum_solver.quantumproblemdefinition`

Status: keep unpublished for now.

Why it is blocked:
- The implementation does not build the advertised Hamiltonian, pulse, or backend-specific simulation objects implied by the quantum-problem contract.

### `molecular_docking.quantum_solver.solutionextraction`

Status: keep unpublished for now.

Why it is blocked:
- The current extraction logic is tied to the placeholder classical solver path and should be re-reviewed only after the quantum-solver contract is repaired or renamed.

Evidence as of 2026-04-19:
- The `pubrev-005` molecular-docking wave advanced only the directly audited safe rows and held the quantum-solver atoms.
- Focused behavior review found placeholder or over-broad classical stand-in behavior for the quantum rows.
- The 2026-04-20 classical remediation wave repaired the greedy mapping, greedy subgraph, and graph-to-UDG rows; the quantum rows remain held.

### `molecular_docking.quantum_solver_d12`

Status: keep the listed rows unpublished for now.

Held atoms:
- `sciona.atoms.bio.molecular_docking.quantum_solver_d12.adiabaticpulseassembler`
- `sciona.atoms.bio.molecular_docking.quantum_solver_d12.interactionboundscomputer`
- `sciona.atoms.bio.molecular_docking.quantum_solver_d12.quantumcircuitsampler`
- `sciona.atoms.bio.molecular_docking.quantum_solver_d12.quantumsolutionextractor`
- `sciona.atoms.bio.molecular_docking.quantum_solver_d12.quantumsolverorchestrator`

Why they are blocked:
- `adiabaticpulseassembler` returns a plain dict stand-in instead of constructing Pulser waveforms, DMM detuning map, channel declarations, pulse objects, and a locked sequence.
- `interactionboundscomputer` returns raw `u_min` and `u_max` values rather than deriving the pulse parameter bounds described by the source evidence.
- `quantumcircuitsampler` runs deterministic classical greedy/simulated-annealing logic instead of executing the advertised qutip, tensor-network/MPS, or state-vector backend path.
- `quantumsolutionextractor` decodes bitstrings into lists, but the source evidence describes node-set solution objects tied to Pulser register mapping.
- `quantumsolverorchestrator` bypasses register assembly, bandwidth optimization, pulse packaging, backend sampling, and solution extraction; it directly runs a classical greedy/SA loop.

Proposed fixes:
1. Implement the end-to-end neutral-atom MWIS pipeline described by the matches/CDG evidence.
2. Add focused behavior tests for pulse assembly, interaction-bound derivation, backend sampling, and register-aware solution extraction.
3. If the current classical helpers are intentional, rename and re-document them as classical approximations before reentering publication review.

Evidence as of 2026-04-19:
- The `pubrev-028` molecular-docking wave held all five `quantum_solver_d12` rows after direct source review.
- The blocker is semantic drift, not just missing uncertainty metadata.

## Physics

### `physics.pasqal.docking.quantum_mwis_solver`

Status: keep unpublished for now.

Why it is blocked:
- The implementation documents itself as a deterministic MWIS heuristic placeholder for combinatorial optimization rather than a faithful quantum/PASQAL solver.
- The public name implies a quantum maximum-weight independent-set solver, but the current behavior is a local heuristic approximation.
- The source-backed implementation path depends on Pulser/emulator runtimes, so this should become an optional-dependency publication lane rather than a dependency-driven deletion.

What we verified:
- The source docstring for `quantum_mwis_solver` explicitly describes it as a placeholder.
- The physics publishability wave intentionally avoided `physics.pasqal` and ratcheted the smaller Tempo `_zero_offset` lane instead.
- Source review of `../ageo-atoms/third_party/Molecular-Docking/src/solver/quantum_solver_molecular.py:270` shows the true solver path uses Pulser register/sequence construction and emulator sampling.

Proposed fixes:
1. Add a provider optional-dependency group for the Pulser/emulator stack required by the source path.
2. Build and test the source-aligned quantum/PASQAL implementation in a local environment with those optional dependencies installed.
3. Keep the current heuristic unpublished under the quantum name; if it remains useful, rename it as a deterministic heuristic with separate metadata.
4. Reenter publication review only after implementation, optional dependency docs, references, and tests align.

Evidence as of 2026-04-19:
- Local source inspection showed the placeholder description in `sciona.atoms.physics.pasqal.docking.quantum_mwis_solver`.
- A remediation worker confirmed the source-aligned path needs `pulser`, `pulser_simulation`, `emu_sv`, and `emu_mps`, which are not in the baseline matcher venv.

## SciPy

### `scipy.sparse_graph`

Status: keep the graph-spectral helpers unpublished for now.

Why it is blocked:
- `graph_fourier_transform`, `inverse_graph_fourier_transform`, and `heat_kernel_diffusion` are local graph-signal helpers rather than direct SciPy sparse-graph APIs.
- Their public names still imply a broader SciPy sparse-graph contract than the current provenance and review evidence justify.

What we verified:
- In the 2026-04-20 remediation pass, `graph_laplacian` was narrowed to a Laplacian-return wrapper over `scipy.sparse.csgraph.laplacian`, and the shortest-path wrappers were aligned to `scipy.sparse.csgraph.shortest_path`.
- `minimum_spanning_tree` was verified against `scipy.sparse.csgraph.minimum_spanning_tree`.
- `graph_fourier_transform`, `inverse_graph_fourier_transform`, and `heat_kernel_diffusion` remain held because they are local graph-signal helpers rather than direct SciPy sparse-graph APIs.
- The `scipy.stats.norm` remediation row was narrowed to the frozen-distribution call path `scipy.stats.norm(loc=loc, scale=scale)` and removed from remediation.

Proposed fixes:
1. For the misleading spectral helpers, either:
   - rename them to match the actual behavior surface, or
   - tighten the implementation and tests until the current names are defensible.
2. Add behavior-level tests against concrete graph-signal expectations before reentering the publication queue.
3. Only restore these atoms to the publishability lane after the misleading subset is resolved.

Suggested remediation order:
1. `graph_fourier_transform`
2. `inverse_graph_fourier_transform`
3. `heat_kernel_diffusion`

Evidence as of 2026-04-20:
- The `pubrev-077` remediation wave promoted only source-aligned SciPy wrappers and left the graph-spectral helpers held.

### SciPy Naming Debt

Status: rename wave started; revisit remaining naming debt deliberately.

Planning note:
- The concrete rename log and remaining touch points now live in `docs/RENAMING_ATOMS.md`.

What changed:
- The interpolate family rename wave was completed on 2026-04-16:
  - `cubicsplinefit` -> `cubic_spline_fit`
  - `rbfinterpolatorfit` -> `rbf_interpolator_fit`
- Shared source, probes, review bundles, manifest rows, and local replay state were updated together for those two wrappers.

What remains:
- Several SciPy wrappers still use upstream-adjacent short names such as `det`, `inv`, `butter`, `firwin`, and `freqz`.
- Those names are not compressed multi-word forms, so they need a policy decision before any rename wave, not ad hoc churn.

Proposed fixes:
1. Define a repo-wide policy for when legacy upstream-style names are acceptable versus when snake_case normalization is required.
2. For unaudited or unpublished atoms, normalize names before publication instead of after.
3. Treat future rename waves as coordinated source/probe/bundle/manifest/replay changes, not one-off wrapper edits.

Suggested remediation order:
1. Decide whether short established upstream names like `det`, `inv`, `freqz`, and `firwin` should be left as-is under the naming policy.
2. Review stale spatial and optimize alias debt recorded in `docs/RENAMING_ATOMS.md`.

Evidence as of 2026-04-16:
- The interpolate compressed-name debt has been removed from the live catalog.
- Remaining SciPy naming questions are now policy and alias-cleanup issues, not the obvious interpolate wrappers.
