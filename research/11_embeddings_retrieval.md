# Research: Embedding Extraction & Similarity Search Atoms

## Goal

Find best-in-class implementations for embedding operations: extraction,
similarity computation, indexing, and query expansion. Target repo: `sciona-atoms-dl`.

## CDG stages this research covers (~15 stages)

- FAISS index construction and search (Facebook Image Similarity)
- Query expansion / database-side augmentation (Facebook Image Similarity)
- Cosine similarity computation (LLM Prompt Recovery, Shopee)
- Embedding delta/difference vectors (LLM Prompt Recovery)
- Sentence embedding extraction concept (Eedi, CommonLit)
- Image embedding extraction concept (Google Universal Image Embedding)
- PCA on embeddings for dimensionality reduction (Google Landmark, UIE)
- Feature aggregation — rolling statistics on embeddings (IEEE CIS Fraud)
- Embedding normalization — L2 normalize (Google Landmark)
- Re-ranking with learned embeddings (OGB WikiKG90M)
- Nearest neighbor search in embedding space (Shopee, Facebook)

## What to research

### 1. Cosine similarity (pairwise and batch)
- `cosine_similarity_matrix(embeddings_a: NDArray, embeddings_b: NDArray) -> NDArray`
- `cosine_similarity_pair(a: NDArray, b: NDArray) -> float`
- L2-normalize then dot product
- Pure numpy, handle zero-norm edge case

### 2. L2 normalization
- `l2_normalize(embeddings: NDArray, axis: int) -> NDArray`
- `embedding / np.linalg.norm(embedding, axis=axis, keepdims=True)`
- Handle zero vectors

### 3. Query expansion (database-side augmentation)
- Average top-K retrieved embeddings with query embedding
- `query_expansion(query: NDArray, retrieved: NDArray, alpha: float) -> NDArray`
- DBA: `expanded = alpha * query + (1-alpha) * mean(top_k_retrieved)`
- Source: Arandjelovic et al., Chum et al.

### 4. Embedding difference / delta vectors
- `embedding_delta(original: NDArray, transformed: NDArray) -> NDArray`
- Simple subtraction to isolate transformation direction
- Used in LLM Prompt Recovery to find the "prompt direction"

### 5. PCA whitening on embeddings
- Fit PCA, project, optionally whiten
- `pca_reduce_embeddings(embeddings: NDArray, n_components: int, whiten: bool) -> NDArray`
- We have PCA atoms in sklearn — research if a specific embedding-oriented
  variant is needed (e.g., PCA + L2 renormalization, common in retrieval)

### 6. FAISS index construction (opaque wrapper)
- FAISS is a compiled library — atom should be interface-level
- `build_flat_index(embeddings: NDArray, metric: str) -> FaissIndex`
- `search_index(index: FaissIndex, queries: NDArray, k: int) -> tuple[NDArray, NDArray]`
- Note: FAISS has C++ internals — treat as opaque with input/output contracts

### 7. Re-ranking by embedding distance
- `rerank_by_distance(query: NDArray, candidates: NDArray, candidate_ids: NDArray, k: int) -> NDArray`
- Sort candidates by distance to query in embedding space

## Research questions

1. For cosine similarity: what is the most numerically stable implementation?
   (Handle zero vectors, float32 precision issues)
2. For query expansion: what are the standard alpha values?
   (Typically 0.5-0.8 for the query weight)
3. For FAISS: should we treat it as opaque or decompose?
   (Recommend: opaque — it's a compiled index. Atom defines the contract.)
4. What contracts are natural? (embeddings 2D, L2-normalized input for
   cosine, k <= num_candidates, PCA n_components <= embedding_dim)
5. Should embedding extraction from pretrained models be an atom?
   (Recommend: opaque wrapper similar to CNN architectures)

## Output format

Concept types: `searching` for retrieval, `dimensionality_reduction` for
PCA/projection, and `analysis` for similarity.

For each candidate atom, provide:
```
Name: cosine_top_k
Description: Return the top-k nearest reference embeddings by cosine similarity.
Source: URL to the best reference implementation, paper, or library source
License: MIT, BSD, Apache-2.0, or public domain; flag any incompatible license
Concept type: searching, dimensionality_reduction, or analysis
Signature: (query_embeddings: NDArray, reference_embeddings: NDArray,
            k: int) -> tuple[NDArray, NDArray]
Pure function boundary: embedding arrays and explicit retrieval parameters in,
  indices and scores out; no model inference side effects, index persistence,
  network calls, GPU state, or file I/O.
Contracts:
  - require: query_embeddings.shape[1] == reference_embeddings.shape[1]
  - require: k > 0
  - ensure: returned indices refer to rows in reference_embeddings
Witness: tiny 2D embedding set with obvious nearest neighbors; verify top-k
  indices and descending scores.
Dependencies: numpy/sklearn preferred; FAISS or Annoy acceptable only when heavy
  dependency and approximate behavior are clearly documented
CDG stages covered: shopee/retrieval, image_similarity/embedding_search, ...
```
