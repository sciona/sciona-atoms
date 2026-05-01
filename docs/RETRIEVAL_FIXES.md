# Retrieval Fixes — Atoms Found by Deep Search but Missed by TF-IDF

**Date**: 2026-04-30
**Source**: Deep search audit of 146 gap stages across 8 atom repos

## Summary

31 atoms exist and directly match CDG stages but are not surfaced by
`find_matching_primitives()`. 15 more have metadata issues that partially
block retrieval. Fixing these closes 31-46 additional gaps without
creating any new atoms.

## Fix categories

### 1. Add aliases to catalog (solves vocabulary mismatch)

These atoms have correct descriptions but use different terminology than
CDG stages. Add aliases via `catalog.add_alias()`:

| CDG stage term | Atom name | Alias to add |
|---------------|-----------|-------------|
| spatial decompression / IDCT | `idct` | "inverse_dct", "spatial_decompression" |
| collaborative filtering / ALS | `als_user_update` | "collaborative_filtering", "matrix_factorization_als" |
| count encoding | `frequency_encode` | "count_encoding", "count_encode" |
| mean encoding | `target_encode` | "mean_encoding", "mean_encode" |
| beam search decoding | `beam_search` | "beam_decode", "sequence_decoding" |
| variance thresholding | `variance_threshold_fit` | "variance_thresholding", "low_variance_filter" |
| CRF / conditional random field | `dense_crf_2d` | "conditional_random_field", "crf_postprocess" |
| query expansion / DBA | `alpha_query_expansion` | "query_expansion", "database_side_augmentation" |
| TTA / test time augmentation | `tta_10crop_average` | "test_time_augmentation" |
| action execution / best arm | `select_best_arm` | "action_execution", "arm_selection" |
| DICOM windowing | `dicom_window` | "dicom_windowing", "hounsfield_window" |
| signal chunking | `sliding_windows` | "signal_chunking", "chunk_signal" |
| feature hashing | `feature_hasher_csr_matrix` | "feature_hashing" |
| object detection / YOLO | `yolo_object_detector` | "object_detection", "yolo_detection" |

### 2. Fix concept_type mismatches

These atoms have incorrect or overly specific concept_types that prevent
the category bonus from helping retrieval:

| Atom | Current concept_type | Should be | Reason |
|------|---------------------|-----------|--------|
| `dense_crf_2d` | analysis | signal_filter | CRF is a filter/smoother on predictions |
| `tokenize` | (check current) | data_extraction | Tokenization is data extraction |
| `mel_filterbank` | signal_filter | signal_transform | Filterbank creation is a transform |
| `tta_10crop_average` | (check current) | sampler | TTA is a sampling/averaging operation |
| `adjacency_smoothing` | (check current) | graph_traversal | GCN smoothing is graph processing |

### 3. Improve atom descriptions

These atoms have descriptions that don't mention the CDG vocabulary:

| Atom | Missing keywords to add to description |
|------|---------------------------------------|
| `build_faiss_flat_ip` | "dense vector indexing", "embedding index" |
| `adjacency_smoothing` | "graph convolution", "GCN", "spectral" |
| `clean_text` | "regex filtering", "text normalization" |
| `entity_embedding_lookup` | "tabular deep learning", "categorical embedding" |
| `item_popularity_decay` | "trend generation", "trending items" |
| `pairwise_ratios` | "cross-asset features", "relative ratios" |

### 4. Composable atom chains (granularity mismatch)

Some CDG stages describe a multi-step operation that maps to a chain of
atoms. The retrieval system returns individual atoms but can't express
"these 3 atoms together match this stage":

| CDG stage | Atom chain |
|-----------|-----------|
| mel spectrogram | `mel_filterbank` → `apply_mel_filterbank` → `log_mel_spectrogram` |
| BIO decode + span extract | `bio_decode` → `char_to_token_offsets` |
| KMeans clustering | `kmeans_plusplus_initialize_dense` → (opaque KMeans) |
| GNSS processing | `correct_clock_bias` → `filter_by_cn0` → `filter_multipath` |

These should be documented as multi-atom bindings in the CDG bindings files
rather than expecting a single atom to match.

## Implementation priority

1. **Aliases** (highest impact, lowest effort) — add to `seed_builtin_primitives()`
   or a new `seed_solution_aliases()` function in catalog.py
2. **Concept type fixes** — update the cdg.json files in the affected atom repos
3. **Description improvements** — edit atom docstrings and cdg.json descriptions
4. **Multi-atom bindings** — update _bindings.json to reference atom chains

## Verification

After fixes, re-run:
```
pytest tests/test_retrieval_solution_cdgs.py -v
python3 /tmp/rematch_all_gaps.py
```

Expected: 31+ additional stages should now appear as correct matches in top-5.
