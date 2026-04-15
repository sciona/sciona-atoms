# CS Atoms Gap Analysis

Status of CS algorithmic families needed for `sciona-atoms-cs`.

## Current State

`sciona-atoms-cs/src/` is **empty** — no atom implementations exist yet.

The 13 expansion test files in `sciona-atoms-cs/tests/` all pass (349 tests) because they
only import from the **matcher** (`sciona.expansion_atoms.runtime_*` and
`sciona.principal.expansion_rules.*`). They do not test core algorithmic atom
implementations.

Four core algorithmic families (sorting, graph traversal, graph shortest paths,
search) already live in `sciona-atoms/src/sciona/atoms/algorithmic/` by design.
They are intentionally in the core provider, not in `sciona-atoms-cs`.

## Families Missing Core Atoms

These have runtime diagnostic atoms and expansion rule sets in the **matcher**
but no core algorithmic atom implementations anywhere. Each needs core atoms
written and placed in `sciona-atoms-cs/src/sciona/atoms/`.

### 1. Dynamic Programming
- **Expansion atoms** (matcher): `detect_table_sparsity`, `prune_infeasible_states`, `compress_dp_table`, `validate_subproblem_overlap`
- **Rule set** (matcher): `DynamicProgrammingExpansionRuleSet`
- **Missing core atoms**: e.g., knapsack, longest common subsequence, edit distance, matrix chain multiplication

### 2. String Matching
- **Expansion atoms** (matcher): `analyze_alphabet_size`, `check_pattern_text_ratio`, `measure_hash_collision_rate`, `validate_failure_function`
- **Rule set** (matcher): `StringMatchingExpansionRuleSet`
- **Missing core atoms**: e.g., KMP, Rabin-Karp, Boyer-Moore, Aho-Corasick

### 3. Greedy Algorithms
- **Expansion atoms** (matcher): `validate_matroid_exchange`, `detect_criterion_ties`, `estimate_solution_quality`, `detect_redundant_feasibility`
- **Rule set** (matcher): `GreedyExpansionRuleSet`
- **Missing core atoms**: e.g., activity selection, Huffman coding (as greedy), fractional knapsack, Kruskal/Prim MST

### 4. Combinatorics (Branch-and-Bound)
- **Expansion atoms** (matcher): `analyze_branching_factor`, `monitor_bound_tightness`, `detect_symmetry`, `check_pruning_effectiveness`
- **Rule set** (matcher): `CombinatoricsExpansionRuleSet`
- **Missing core atoms**: e.g., branch-and-bound solver, backtracking, permutation generator, subset enumeration

### 5. Number Theory
- **Expansion atoms** (matcher): `validate_input_range`, `monitor_gcd_convergence`, `check_small_prime_divisors`, `detect_modular_overflow`
- **Rule set** (matcher): `NumberTheoryExpansionRuleSet`
- **Missing core atoms**: e.g., GCD/extended-GCD, modular exponentiation, Miller-Rabin primality, sieve of Eratosthenes

### 6. Geometry (Computational)
- **Expansion atoms** (matcher): `detect_collinear_points`, `analyze_numeric_precision`, `detect_duplicate_points`, `validate_convexity`
- **Rule set** (matcher): `GeometryExpansionRuleSet`
- **Missing core atoms**: e.g., convex hull, line segment intersection, closest pair, point-in-polygon

### 7. Compression
- **Expansion atoms** (matcher): `analyze_compression_ratio`, `validate_lossless_roundtrip`, `detect_dictionary_bloat`, `monitor_encoding_throughput`
- **Rule set** (matcher): `CompressionExpansionRuleSet`
- **Missing core atoms**: e.g., Huffman coding, LZ77, arithmetic coding, run-length encoding

### 8. Information Theory
- **Expansion atoms** (matcher): `check_distribution_support`, `analyze_sample_sufficiency`, `detect_numerical_underflow`, `validate_information_inequality`
- **Rule set** (matcher): `InformationTheoryExpansionRuleSet`
- **Missing core atoms**: e.g., Shannon entropy, KL divergence, mutual information, channel capacity

### 9. Randomized Algorithms
- **Expansion atoms** (matcher): `validate_hash_independence`, `analyze_sketch_accuracy`, `monitor_sample_coverage`, `check_concentration_bound`
- **Rule set** (matcher): `RandomizedExpansionRuleSet`
- **Missing core atoms**: e.g., randomized quicksort, reservoir sampling, count-min sketch, bloom filter

## Families in Matcher with No CS Repo Tests

These expansion families exist in the matcher but have **no corresponding test
files** in `sciona-atoms-cs`. They may belong in other provider repos or may
need CS tests added:

| Family | Likely Provider Repo |
|--------|---------------------|
| Neural Network | `sciona-atoms-ml` |
| Dimensionality Reduction | `sciona-atoms-ml` |
| Clustering | `sciona-atoms-ml` |
| Signal Detect/Measure | `sciona-atoms-signal` |
| Signal Filter | `sciona-atoms-signal` |
| Signal Transform | `sciona-atoms-signal` |
| Graph Signal Processing | `sciona-atoms-signal` or `sciona-atoms` |
| Kalman Filter | `sciona-atoms` (exists) |
| Particle Filter | `sciona-atoms` (exists) |
| Sequential Filter | `sciona-atoms` (exists) |
| Belief Propagation | `sciona-atoms` (exists) |
| Linear Algebra | `sciona-atoms` (scipy/numpy) |
| Optimization | `sciona-atoms` (scipy) |
| Quadrature | `sciona-atoms` (scipy) |
| ODE Solver | `sciona-atoms` (scipy) |
| MCMC | `sciona-atoms` (inference) |
| VI/ADVI | `sciona-atoms` (inference) |

## Implementation Plan

Priority order (by foundational importance):
1. Dynamic Programming (5-6 atoms)
2. String Matching (4 atoms)
3. Number Theory (4-5 atoms)
4. Greedy Algorithms (4 atoms)
5. Combinatorics/Backtracking (4 atoms)
6. Computational Geometry (4-5 atoms)
7. Compression (4 atoms)
8. Information Theory (4 atoms)
9. Randomized Algorithms (4 atoms)

Each family needs: atom module, witnesses, probes, heuristic metadata JSON, and import smoke tests.
