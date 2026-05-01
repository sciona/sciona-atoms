# Research: Recommender System Primitive Atoms

## Goal

Find best-in-class, pure-function implementations for recommender system
building blocks: candidate generation, collaborative filtering components,
and ranking features. Target repo: `sciona-atoms-dl`.

## CDG stages this research covers (~15 stages)

- Co-visitation / co-occurrence matrix (OTTO Recommender)
- Candidate generation from co-occurrence (OTTO Recommender)
- ALS matrix factorization components (H&M Fashion, PetFinder)
- Popularity-based candidate generation (H&M Fashion)
- Session-based features (OTTO Recommender)
- User-item interaction features (H&M Fashion, Instacart)
- Item trending / recency scoring (H&M Fashion)
- Negative sampling — uniform and in-batch (existing atoms, check coverage)
- Re-ranking by feature combination (OTTO Recommender)
- User purchase history aggregation (Instacart, H&M Fashion)
- Candidate merging from multiple retrieval channels (H&M Fashion)
- Click-through rate features (Outbrain Click Prediction)

## What to research

### 1. Co-visitation / co-occurrence matrix
- Count how often items appear together in sessions
- `co_occurrence_matrix(sessions: list[list[int]], n_items: int, time_weights: NDArray | None) -> NDArray`
- Sparse implementation for large item catalogs
- Source: competition kernels (OTTO Recommender top solutions)

### 2. Candidate generation from co-occurrence
- For a given user's recent items, retrieve top-K co-occurring items
- `cooccurrence_candidates(user_items: list[int], cooccurrence: NDArray, k: int) -> NDArray`
- Sum co-occurrence scores across user's items, return top-K

### 3. ALS update step (pure numpy)
- Alternating Least Squares for matrix factorization
- `als_user_update(ratings: NDArray, item_factors: NDArray, regularization: float) -> NDArray`
- `als_item_update(ratings: NDArray, user_factors: NDArray, regularization: float) -> NDArray`
- The closed-form solution: `U = (V^T V + λI)^{-1} V^T R`
- Source: implicit library (MIT), or pure numpy implementation

### 4. Popularity scoring
- `item_popularity(interactions: NDArray, time_col: NDArray, decay: float) -> NDArray`
- Count interactions with optional time decay
- Simple aggregation — most popular items as candidates

### 5. Session features
- `session_features(item_sequence: list[int], timestamps: NDArray) -> dict`
- Session length, unique items, time span, inter-action gaps
- Pure Python/numpy

### 6. User-item interaction features
- `user_item_features(user_history: NDArray, item_id: int) -> dict`
- Has user seen this item before, days since last interaction,
  number of interactions, average rating
- Pure aggregation operations

### 7. Reciprocal rank fusion
- Merge ranked lists from multiple retrieval channels
- `reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int) -> list[int]`
- Score = sum(1 / (k + rank_in_list)) across lists
- Source: Cormack et al. 2009

## Research questions

1. For co-occurrence: what is the memory-efficient sparse implementation?
   (scipy.sparse COO/CSR for large catalogs)
2. For ALS: should we implement the full loop or just one update step?
   (One step — the loop is orchestration)
3. For popularity: what time decay functions are used in practice?
   (Exponential decay with configurable half-life)
4. What contracts are natural? (co-occurrence matrix symmetric,
   ALS factors have matching dimensions, k > 0)
5. How do we handle cold-start? (Document as precondition —
   user must have at least 1 interaction)

## Output format

Concept types: `searching` for candidate generation, `data_assembly` for feature
computation, and `optimization` for ALS.

For each candidate atom, provide:
```
Name: item_item_candidates
Description: Generate recommendation candidates from item-item co-occurrence
  scores for a user's interaction history.
Source: URL to the best reference implementation, paper, or library source
License: MIT, BSD, Apache-2.0, or public domain; flag any incompatible license
Concept type: searching, data_assembly, or optimization
Signature: (user_items: NDArray, item_similarity: NDArray, k: int) -> NDArray
Pure function boundary: interaction arrays, similarity matrices, and explicit
  parameters in, candidate IDs or score arrays out; no database access, model
  training side effects, global state, or file I/O.
Contracts:
  - require: k > 0
  - require: user_items contains valid item indices
  - ensure: returned candidates exclude already-seen items when specified
Witness: small item similarity matrix and two user history items; verify ranked
  candidates and exclusion behavior.
Dependencies: numpy/scipy preferred; implicit/sklearn acceptable when justified
CDG stages covered: h_and_m/candidate_generation, otto/session_retrieval, ...
```
