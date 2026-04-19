# REMEDIATION

This file tracks heavier-lift catalog debt that should not be papered over with relaxed publishability rules. These items need real semantic repair, better tests, or clearer scope before they should be promoted into the public catalog.

## Inference

### `mcmc_foundational.mini_mcmc` sampling and NUTS transition rows

Status: keep the listed loop/tree transition rows unpublished for now.

Held atoms:
- `sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc.metropolishmctransition`
- `sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc.runsamplingloop`
- `sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc_llm.collectposteriorchain`
- `sciona.atoms.inference.mcmc_foundational.mini_mcmc.nuts_llm.runnutstransitions`
- `sciona.atoms.inference.mcmc_foundational.mini_mcmc.nuts.nuts_recursive_tree_build`

Why they are blocked:
- `metropolishmctransition` advertises a full HMC transition that samples momentum, invokes leapfrog, accepts/rejects, and returns an updated state. The implementation only consumes a precomputed proposal, samples an original momentum through hidden seed coupling, and preserves the previous gradient even after accepted moves.
- `runsamplingloop` advertises repeated HMC transitions, but the current loop never calls a transition kernel and does not update the chain position.
- `collectposteriorchain` advertises transition-kernel collection, but the current loop records the unchanged chain state and only advances an RNG seed.
- `runnutstransitions` advertises NUTS transitions, but the current implementation performs a simple random-walk position update with no log-probability, leapfrog tree building, slice variable, U-turn criterion, divergence logic, or Metropolis correction.
- `nuts_recursive_tree_build` advertises a NUTS trajectory object, but it returns only a rightmost position array and drops the trajectory metadata required for a semantically valid NUTS tree builder.

What we verified:
- The defensible mini-MCMC subset is limited to initialization helpers, the pure leapfrog proposal kernel, and the combined `hmc_llm.hamiltoniantransitionkernel`.
- References were corrected to current `sciona.atoms.inference...` runtime FQDNs, with low-confidence conceptual-only attribution for held rows.
- Focused pubrev-008 tests cover the safe subset and assert that held rows are absent from the catalog-ready review bundle.

Proposed fixes:
1. Replace `metropolishmctransition` with a contract that either accepts both original and proposed momenta explicitly, or make it a complete transition that samples momentum, invokes leapfrog internally, recomputes the accepted gradient, and returns consistent diagnostics.
2. Replace `runsamplingloop` and `collectposteriorchain` with loops that accept a transition callable or log-probability oracle and actually thread state through repeated transitions before collecting samples.
3. Replace `runnutstransitions` with a real NUTS transition implementation using slice variables, recursive tree expansion, no-u-turn checks, divergence checks, and acceptance-statistic accumulation.
4. Replace `nuts_recursive_tree_build` with a structured trajectory return type containing left/right states, candidate proposal, valid count, stop/divergence flags, and acceptance statistics.
5. Add behavior-level tests on a small Gaussian target before reentering publication review.

Evidence as of 2026-04-19:
- The pubrev-008 review held these rows instead of forcing publication through family-level MCMC metadata.
- Local source inspection showed placeholder or semantically incomplete loop/tree behavior for the held atoms.

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

### `e2e_ppg.kazemi_wrapper.wrapperpredictionsignalcomputation`

Status: keep unpublished for now.

Why it is blocked:
- The current implementation returns `prediction * raw_signal`, which is a simple elementwise multiplication rather than the upstream Kazemi peak-detection post-processing behavior described by the existing metadata.
- The surrounding `e2e_ppg.kazemi_wrapper` family has valid publication candidates, but this row has semantic drift and should not be promoted by shared family metadata.
- Publishing it under the current name would imply a stronger upstream wrapper contract than the implementation provides.

What we verified:
- The `signalarraynormalization` sibling can be reviewed separately because it delegates to the upstream `kazemi_peak_detection.normalize` routine.
- `wrapperpredictionsignalcomputation` does not currently delegate to the upstream Kazemi wrapper path and only combines arrays locally.

Proposed fixes:
1. Replace `wrapperpredictionsignalcomputation` with behavior equivalent to the upstream Kazemi wrapper routine, or rename and narrow the public contract to the local elementwise operation.
2. Add behavior-level tests against representative upstream-shaped prediction and raw-signal inputs.
3. Reenter publication review only after the source behavior, references, CDG, and review bundle all describe the same contract.

Evidence as of 2026-04-19:
- The e2e-PPG publishability review wave held this atom while approving the safe heart-cycle and normalization rows.
- Local source inspection showed the implementation returns `prediction * raw_signal`.

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

### `sciona.atoms.bio.mint.axial_attention` RowSelfAttention rows

Status: keep the listed axial-attention rows unpublished for now.

Held atoms:
- `sciona.atoms.bio.mint.axial_attention.row_self_attention`
- `sciona.atoms.bio.mint.axial_attention.rowselfattention`

Why they are blocked:
- The current wrappers are generic 2D/3D scaled dot-product attention approximations, not source-aligned MINT `RowSelfAttention`.
- Upstream MINT uses a parameterized `torch.nn.Module` over 4D MSA tensors with `q_proj`, `k_proj`, `v_proj`, `out_proj`, row aggregation, and different padding/mask semantics.
- The NumPy row has no upstream MINT callable to substantiate a MINT RowSelfAttention claim.

Proposed fixes:
1. Replace the torch row with a source-aligned RowSelfAttention module/core API that accepts or owns projection parameters and 4D MSA tensors.
2. Add behavior tests for upstream shape contracts, projection behavior, row aggregation, padding masks, and batched forward behavior.
3. Either remove/re-scope the NumPy row as generic attention with independent provenance, or provide a clearly documented source-aligned NumPy replay that does not claim to be upstream MINT RowSelfAttention.

Evidence as of 2026-04-19:
- The `pubrev-019` MINT review held these rows after direct comparison with upstream `mint/axial_attention.py` at commit `79a9956b80e221066a3007bb156e8e986cba6190`.
- Provider review tests assert both rows remain remediation-only while the safe MINT subset advances.

### `molecular_docking.greedy_mapping_d12.construct_mapping_state_via_greedy_expansion`

Status: keep unpublished for now.

Why it is blocked:
- The current implementation uses a simplified first-free-site placement path rather than the advertised D12/lattice placement semantics.
- Publication should wait for explicit feasibility checks, deterministic seed handling, and preservation of the immutable mapping-state contract.

### `molecular_docking.greedy_mapping_d12.orchestrate_generation_and_validate`

Status: keep unpublished for now.

Why it is blocked:
- The current wrapper validates a supplied mapping state but does not drive iterative generation from `starting_node` or invoke the expansion stages implied by its public contract.

### `molecular_docking.greedy_subgraph.greedy_maximum_subgraph`

Status: keep unpublished for now.

Why it is blocked:
- The implementation greedily selects a high-score independent set by treating adjacency as conflicts.
- The public name and metadata imply a connected maximum-weight subgraph selection contract that the current behavior does not establish.

### `molecular_docking.map_to_udg.graphtoudgmapping`

Status: keep unpublished for now.

Why it is blocked:
- The current spectral-layout heuristic can add edges and does not prove faithful mapping of the input graph into a unit-disk graph representation.
- Publication needs either a semantically explicit UDG embedding contract or implementation changes that prove edge preservation.

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
- The `pubrev-005` molecular-docking wave advanced only the directly audited safe rows and held these seven atoms.
- Focused behavior review found placeholder, over-broad classical stand-in, or semantic-drift behavior for the held rows.

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
- The family also contains related metadata gaps, so forcing the Pasqal lane through review would hide a real semantic mismatch.

What we verified:
- The source docstring for `quantum_mwis_solver` explicitly describes it as a placeholder.
- The physics publishability wave intentionally avoided `physics.pasqal` and ratcheted the smaller Tempo `_zero_offset` lane instead.

Proposed fixes:
1. Decide whether the atom should expose a true PASQAL/quantum MWIS primitive or be renamed as a deterministic heuristic helper.
2. If retained as a quantum solver, replace the placeholder with behavior grounded in the intended solver path and add behavior-level tests.
3. If retained as a heuristic, rename and remap the metadata so the public contract no longer implies quantum solver parity.
4. Reenter publication review only after the implementation, name, references, and tests align.

Evidence as of 2026-04-19:
- Local source inspection showed the placeholder description in `sciona.atoms.physics.pasqal.docking.quantum_mwis_solver`.
- The lane was held during the parallel publishability wave rather than being forced through bundle review.

## SciPy

### `scipy.sparse_graph`

Status: keep unpublished for now.

Why it is blocked:
- The remaining reviewed atoms in this family are not just missing audit metadata; they are currently flagged as misleading in the audit backlog.
- The family mixes true missing-row constructor/wrapper work with higher-risk graph-spectral helpers whose public names imply standard SciPy graph operations more cleanly than the current local semantics justify.
- This is not a good candidate for ratcheting by bundle-only review. It needs semantic narrowing or behavior repair first.

What we verified:
- As of 2026-04-16, the backlog still marks `graph_fourier_transform`, `graph_laplacian`, `heat_kernel_diffusion`, and `inverse_graph_fourier_transform` as `misleading`.
- The remaining `all_pairs_shortest_path`, `minimum_spanning_tree`, and `single_source_shortest_path` rows are still `missing_row`, so the family is split between naming/metadata debt and semantic review debt.
- Because of that split, forcing the whole family through publishability review would blur together two different remediation classes.

Proposed fixes:
1. Separate the family into:
   - faithful sparse-graph wrappers that can eventually be published once canonical rows and metadata exist
   - graph-spectral helpers whose current names or semantics are too misleading for publication
2. For the misleading spectral helpers, either:
   - rename them to match the actual behavior surface, or
   - tighten the implementation and tests until the current names are defensible.
3. Add behavior-level tests against concrete SciPy sparse/csgraph expectations before reentering the publication queue.
4. Only restore these atoms to the publishability lane after the misleading subset is resolved.

Suggested remediation order:
1. `graph_laplacian`
2. `graph_fourier_transform`
3. `inverse_graph_fourier_transform`
4. `heat_kernel_diffusion`
5. Reevaluate whether the path/minimum-spanning-tree rows should be treated as clean missing-row publication work in a separate wave.

Evidence as of 2026-04-16:
- Local matcher backlog refresh after the SciPy ratchet waves still shows the four spectral helpers as `misleading`.
- The remaining path/tree atoms are still blocked as `missing_row` rather than review-approved publication candidates.

### `scipy.stats.norm`

Status: keep unpublished for now.

Why it is blocked:
- The current backlog still marks this wrapper as misleading rather than simply unaudited.
- Publishing a distribution-constructor atom under a misleading signature or semantics surface would create a poor public contract because the returned object carries more behavior than the audit surface currently justifies.

What we verified:
- As of 2026-04-16, `sciona.atoms.scipy.stats.norm` remains non-publishable with verdict `misleading`.
- The rest of the `scipy.stats` family was publishable without it, so there was no reason to lower the bar and force `norm` through the ratchet.

Proposed fixes:
1. Re-audit the exact public contract of `norm`, including constructor signature, returned object semantics, and whether the atom should expose a frozen distribution object at all.
2. Add behavior-level tests that validate the intended callable/returned-object surface against upstream SciPy.
3. If the returned frozen-distribution contract is too broad for a public atom, replace it with narrower explicit primitives instead of publishing the current wrapper as-is.

Suggested remediation order:
1. Decide whether `norm` should remain a first-class atom.
2. If yes, tighten its contract and tests.
3. Reenter publication review only after the misleading-status cause is resolved.

Evidence as of 2026-04-16:
- Local backlog refresh after the SciPy ratchet waves still shows `sciona.atoms.scipy.stats.norm` as `misleading`.

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
