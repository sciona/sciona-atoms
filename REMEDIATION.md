# REMEDIATION

This file tracks heavier-lift catalog debt that should not be papered over with relaxed publishability rules. These items need real semantic repair, better tests, or clearer scope before they should be promoted into the public catalog.

## Inference

### `mcmc_foundational.kthohr_mcmc` remaining advanced MCMC rows

Status: keep the listed KTHOHR MCMC rows unpublished for now.

Held atoms:
- `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.nuts.nuts_recursive_tree_build`
- `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.rmhmc.buildrmhmctransitionkernel`

Why they are blocked:
- `nuts.nuts_recursive_tree_build` still lacks source-shaped structured trajectory bookkeeping such as left/right states, candidate, stop flag, valid/divergence counts, and acceptance statistics.
- `rmhmc.buildrmhmctransitionkernel` still lacks the source RMHMC tensor-derivative callback and implicit generalized leapfrog machinery.
- These rows need either optional compiled KTHOHR bindings or a deliberately renamed/narrowed educational lane with focused behavior tests.

Proposed fixes:
1. Rebuild NUTS with a structured trajectory result and tests that validate tree bookkeeping.
2. Implement an optional compiled/FFI lane for RMHMC, or rename it as a limited educational approximation with metric-derivative tests.
3. Reenter publication review only after implementation, metadata, references, and behavior tests align.

Evidence as of 2026-04-20:
- The KTHOHR remediation wave repaired AEES, DE, HMC, MALA, dispatcher, and RWMH rows as explicit local NumPy/educational kernels with limitations.
- The same wave kept NUTS and RMHMC in remediation because the remaining source contracts are materially broader than the current local implementations.

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
