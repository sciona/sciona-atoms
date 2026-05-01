# CDG Grounding Audit — Release Candidate Readiness

**Date**: 2026-04-30 (post-research re-audit; initial 2026-04-24, v2 2026-04-25)
**Auditor**: Claude (automated analysis + 5-worker semantic verification x2)
**Scope**: All 125 CDG templates in `sciona-atoms/data/solution_cdgs/`

## Executive Summary

125 CDG templates covering 100+ Kaggle/benchmark problem types. **7 are fully
grounded** (every stage bound to an existing atom). 409 unbound stages across
117 CDGs were verified against the 1,837-atom catalog (up from 1,360 pre-research).

### Post-research verification (2026-04-30)

| Verdict | Pre-research (Apr 25) | Post-research (Apr 30) | Delta |
|---------|----------------------|----------------------|-------|
| **correct** | 18 (4.4%) | 108 (26.4%) | **+90** |
| **partial** | 74 (18.1%) | 121 (29.6%) | **+47** |
| **false_positive** | 317 (77.5%) | 180 (44.0%) | **-137** |

16 deep research projects across image augmentation, loss functions, detection
post-processing, text/NLP, audio/speech, medical imaging, CNN architectures,
video/temporal, graph construction, gradient boosting, embeddings, segmentation,
recommender systems, time series, geospatial, and Bayesian bandits produced
**477 new atoms**, closing 137 gaps. The correct+partial rate went from 22.5%
to **56%** of all gap stages.

**180 stages remained as false positives** after initial verification. A deep
search (grepping through actual atom repos, not just the retrieval index)
found 31 additional atoms that exist but retrieval missed due to vocabulary,
concept_type, or granularity mismatches. 15 more have metadata issues that
partially block retrieval. Combined with 34 generic data ops reclassifiable
as orchestration/trivial, the **true remaining gap is 100 stages**.

---

## Retrieval Scoring Redesign

The original keyword overlap scoring was redesigned on 2026-04-24. Changes:

1. **TF-IDF weighted overlap** — rare terms like "decorrelation", "quantile"
   score 3-5x more than generic terms like "compute", "data", "fit"
2. **Extended stop words** — 60+ domain-generic terms filtered (was 15)
3. **IO port name matching** — Jaccard overlap of port names, max +2.0 bonus
4. **Category proximity groups** — exact match +1.5, same-group +0.5, else 0.0

**Test suite impact**: 82 passed, 1 xfailed (was 80 passed, 2 failed, 1 xfailed).
The `pinball_loss` retrieval failure is now fixed.

File: `sciona-matcher/sciona/architect/catalog.py`

---

## Verified Matches (Post-Research, 2026-04-30)

### 108 Correct Matches (bind immediately)

These atoms directly implement the CDG stage. 108 stages across 60+ CDGs.

**Most reused correct-match atoms:**

| Atom | Stages | Research domain |
|------|--------|----------------|
| `tta_geometric_average` | 3 | Image augmentation |
| `tfidf_vectorizer_transform` | 3 | Text/NLP (pre-existing) |
| `resnet_family_backbone` | 3 | CNN architectures |
| `efficientnet_backbone` | 3 | CNN architectures |
| `recurrent_sequence_model` | 3 | CNN architectures |
| `stack_adjacent_frames` | 3 | Video/temporal |
| `nms` | 3 | Detection post-processing |
| `median_filter_1d` | 2 | Audio/speech |
| `qwk_loss` | 2 | Loss functions |
| `cutmix_apply` | 2 | Image augmentation |
| `log_mel_spectrogram` | 2 | Audio/speech |
| `connected_components` | 2 | Graph construction |
| `extract_pseudo_labels` | 2 | Text/NLP |
| `snap_to_nearest` | 2 | Geospatial/sensors |
| `rolling_window_features` | 2 | Time series |
| `cosine_similarity_matrix` | 2 | Embeddings/retrieval |
| `fold_ensemble_average` | 2 | Image augmentation |
| `extract_25d_slices` | 2 | Medical imaging |

**Coverage by research domain:**

| Domain | Correct atoms landed | Example matches |
|--------|---------------------|-----------------|
| Image augmentation | cutmix, TTA, ben_graham, resize_and_pad | 8 stages |
| Loss functions | qwk, ctc, lovasz, crps, focal, weighted_bce, multimodal_nll, weighted_multitask | 10 stages |
| Detection post-processing | nms, nms_1d, wbf, wbf_1d, masks_to_boxes, threshold_detections | 9 stages |
| Audio/speech | log_mel, resample, median_filter, g2p, dtw, whisper, ctc_decode | 8 stages |
| CNN architectures | efficientnet, resnet, swin, yolo, unet_1d, slowfast, rnn, transformer | 12 stages |
| Video/temporal | uniform_sample, sample_frame, stack_adjacent, temporal_mean_pool | 7 stages |
| Graph construction | molecular_distance_graph, co_occurrence_matrix, connected_components, wkt_to_mask | 6 stages |
| Gradient boosting/tabular | target_encode, pairwise_products, group_aggregate, rolling_window | 7 stages |
| Embeddings/retrieval | cosine_similarity, embedding_delta, faiss, pca_whiten_reduce | 5 stages |
| Segmentation/morphology | morphological_close, filter_components_by_area | 3 stages |
| Recommender systems | co_occurrence_matrix, cooccurrence_candidates | 3 stages |
| Time series | rolling_window_features, create_lag_features, sliding_windows, technical_indicators | 5 stages |
| Geospatial/sensors | rts_smooth, snap_to_nearest, detect_steps, filter_multipath | 5 stages |
| Bayesian bandits | initialize_beta_beliefs, bayesian_ucb, beta_bernoulli_update | 3 stages |
| Pre-existing atoms | tfidf, ranked_prediction_blend, voting, ensemble_logit, pinball_loss, etc. | 17 stages |

### 74 Partial Matches (atom covers a component)

These atoms implement a building block of the stage but need wrapping
or combining with other atoms:

**Most reused partial-match atoms:**

| Atom | Stages | What it covers |
|------|--------|---------------|
| `cross_validation` | 3 | CV fold splitting for training stages |
| `auxiliary_logit_loss_fusion` | 3 | Weighted multi-head loss combination |
| `lung_mask_with_bone_removal` | 3 | 3D morphological ops on binary masks |
| `stacking_meta_feature_matrix` | 3 | Feature concatenation from multiple sources |
| `extract_patches_2d` | 2 | Image patch/tile extraction |
| `reconstruct_from_patches_2d` | 2 | Patch-to-image stitching |
| `entity_embedding_lookup` | 2 | Categorical embedding lookup + concatenation |
| `mlp_forward_pass` | 2 | Dense layer forward pass (classification heads) |
| `networkx_weighted_graph_materialization` | 2 | Weighted graph construction from edges |
| `patch_extractor_transform` | 2 | Image patch extraction |
| `ranking_moments_extractor` | 2 | Rank-based feature statistics |
| `select_from_model_threshold` | 2 | Feature selection threshold resolution |
| `assemble_weighted_interaction_graph` | 2 | NetworkX graph assembly from edges |

Notable partial matches:
- `gp_train_cholesky` for GP interpolation (PLAsTiCC light curves)
- `ransac_consensus_is_better` for fundamental matrix estimation
- `invariant_point_attention` for molecular attention mechanisms
- `graph_laplacian` for GCN preprocessing
- `bandpass_filter` for wavelet denoising
- `grid_to_graph` for 2D grid-to-graph conversion (ARC)
- `hashing_vectorizer_token` for feature hashing (Outbrain)

### 180 Remaining False Positives (post-research)

Down from 317 pre-research. These stages still have no useful atom match.

**By domain theme** (from post-research verification):

| Theme | Stages | What's still needed |
|-------|--------|---------------------|
| Generic data ops | ~30 | Concat, split, reshape — too generic for keyword matching |
| Image-specific | ~22 | Face detection, mosaic augment, stain translation |
| Transformer/BERT | ~14 | Pre-trained NLP model inference wrappers |
| Data loading/parsing | ~10 | BSON, DICOM crop, GNSS log, proprietary formats |
| LightGBM/XGBoost | ~8 | Gradient boosting training as opaque atoms |
| Tokenization | ~6 | Subword tokenizers (BPE, WordPiece) |
| GNN message passing | ~5 | Graph neural network layer atoms |
| Domain-specific | ~85 | Competition-specific feature engineering, routing logic |

---

## Current State

### CDG Coverage

| Metric | Count |
|--------|-------|
| Total CDG templates | 125 |
| CDGs with bindings files | 100 |
| CDGs without any bindings | 25 |
| **Fully grounded CDGs** | **7** |
| CDGs with at least 1 verified correct/partial match | 56 |

### Fully Grounded CDGs

1. `adversarial_attacks_1st` — 7 stages, sciona-atoms-ml + sciona-atoms-dl
2. `barachant_seizure_1st` — 7 stages, sciona-atoms-signal
3. `cause_effect_2nd` — 4 stages, sciona-atoms
4. `connectomics_1st` — 5 stages, sciona-atoms-bio
5. `dsb2017_1st` — 8 stages, sciona-atoms-dl
6. `flavours_physics_1st` — 5 stages, sciona-atoms-ml
7. `trackml_5th` — 7 stages, sciona-atoms-physics

### Stage Breakdown (all 125 CDGs)

| Category | Count | % |
|----------|-------|---|
| Total stages | ~697 | 100% |
| Bound active (from 7 grounded CDGs + partial bindings) | 82 | 11.8% |
| Orchestration (no atom needed) | 206 | 29.6% |
| **Verified correct match (post-research)** | **108** | **15.5%** |
| **Verified partial match** | **121** | **17.4%** |
| **Verified false positive (remaining gaps)** | **180** | **25.8%** |

### Atom Library Inventory

| Repository | Pre-research | Post-research | Delta | Focus |
|-----------|-------------|--------------|-------|-------|
| sciona-atoms-ml | 752 | 1,057 | +305 | sklearn decomposition, tabular, feature engineering |
| sciona-atoms | 183 | 208 | +25 | Foundation, graph construction, Bayesian bandits |
| sciona-atoms-dl | 25 | 119 | +94 | DL training, losses, augmentation, detection, video, embeddings |
| sciona-atoms-signal | 92 | 119 | +27 | Audio/speech, time series, anomaly detection |
| sciona-atoms-physics | 102 | 102 | 0 | Particle tracking, pulsars, astronomical, quantum |
| sciona-atoms-fintech | 78 | 78 | 0 | Quantitative finance, derivatives, trading |
| sciona-atoms-bio | 64 | 74 | +10 | Molecular docking, medical imaging, AlphaFold |
| sciona-atoms-robotics | 52 | 52 | 0 | PRONTO state estimation, kinematics, controls |
| sciona-atoms-geo | 4 | 18 | +14 | Geospatial, GNSS, sensor fusion |
| sciona-atoms-cs | 8 | 10 | +2 | Game AI, metaheuristics, dynamic programming |
| **Total** | **1,360** | **1,837** | **+477** | |

### Retrieval Test Results

- **83 tests**, 82 passed, 1 xfail
- **Recall@5**: 100% excluding known xfail
- **Known xfail**: `dsb2017_1st/noisy_or_pooling` — needs embedding path
- PCA binding updated: `pca_fit` → `pca_whiten_reduce` (better match for embedding pipeline)
- Test file: `sciona-matcher/tests/test_retrieval_solution_cdgs.py`

---

## Path to RC1

### Phase 1: Bind verified matches (18 stages, immediate)

Bind the 18 correct matches listed above. This gives 8 more CDGs at
least one new active binding. No new atoms needed.

### Phase 2: Promote partial matches where possible (~30 stages)

For the 74 partial matches, determine which can be promoted to active
bindings with minor adaptation notes. Focus on the most reused atoms:
`stacking_meta_feature_matrix` for feature concatenation,
`cross_validation` for CV stages, `extract_patches_2d` for tiling.

### Phase 3: Mark architectures as opaque (~25 stages)

Reclassify CNN/transformer/pretrained-model stages as `is_opaque=true`:
- CNN backbones (EfficientNet, ResNet, Swin, DenseNet)
- Language models (BERT, DeBERTa, Whisper, Code Llama)
- Specialized architectures (U-Net, YOLO, GRU, protein transformers)

These are pretrained model choices, not algorithmic atoms to decompose.

### Phase 4: Standard atom batch creation (~100-120 stages)

New atoms needed in well-defined categories, suitable for parallel
Codex ingestion:

**Tier 1 — High leverage (unblocks 5+ CDGs each):**
- Image augmentation atoms (CutMix, MixUp, geometric transforms, TTA)
- Feature concatenation/stacking (horizontal concat, sparse stacking)
- Standard losses (CTC, focal, Lovasz, CRPS, QWK, contrastive)
- Post-processing (median filter, NMS, WBF, threshold smoothing)
- Text tokenization/encoding (BPE, WordPiece, label encoding)

**Tier 2 — Medium leverage (unblocks 2-4 CDGs):**
- Audio preprocessing (mel spectrogram, resampling, normalization)
- Medical imaging (DICOM loading, windowing, 3D interpolation)
- Embedding wrappers (sentence-transformer, image embedding extraction)
- Object detection utilities (anchor generation, box encoding/decoding)
- Graph construction (from adjacency, from coordinates, from edges)

**Tier 3 — Single-CDG needs:**
- Domain-specific atoms (forced alignment, stain normalization, GNSS
  parsing, seismic features, etc.)

### Phase 5: Domain-specific tail (~80-100 stages)

Prioritize by CDG importance:
- Recommender systems (co-visitation, ALS, candidate retrieval)
- Video processing (frame extraction, temporal aggregation, tracking)
- Bayesian/bandit (Beta-UCB, Thompson sampling, belief updating)
- Reinforcement learning (game AI state, MCTS components)

### Phase 6: Create bindings for all 125 CDGs

After phases 1-5, create/update bindings for every CDG to reflect the
expanded atom library.

### Estimated Atom Creation Effort

| Category | New atoms | Complexity |
|----------|----------|------------|
| Image/augmentation | ~30 | Standard — well-defined wrappers |
| Text/NLP | ~25 | Standard — tokenizer/encoder wrappers |
| Loss functions | ~15 | Standard — mathematical definitions |
| Post-processing | ~15 | Standard — filter/threshold operations |
| Embedding wrappers | ~15 | Standard — model inference wrappers |
| Object detection | ~10 | Medium — geometric operations |
| Medical imaging | ~10 | Medium — format-specific I/O |
| Audio/speech | ~10 | Medium — signal processing |
| Graph construction | ~8 | Standard — adjacency/edge operations |
| Recommender | ~8 | Medium — interaction patterns |
| Video | ~6 | Medium — temporal operations |
| Tabular ML | ~6 | Standard — LightGBM/XGB wrappers |
| Domain-specific | ~50 | Complex — requires domain expertise |
| **Total** | **~208** | |

### Estimated Impact

| Phase | Stages resolved | Cumulative grounded CDGs |
|-------|----------------|-------------------------|
| Current | 0 | 7 / 125 (5.6%) |
| Phase 1 (bind verified) | 18 | ~10 |
| Phase 2 (promote partial) | ~30 | ~15 |
| Phase 3 (opaque arch) | ~25 | ~25 |
| Phase 4 (standard atoms) | ~100-120 | ~60-70 |
| Phase 5 (domain tail) | ~80-100 | ~100-110 |
| Phase 6 (bind all) | remaining | ~110-120 |

---

## Appendix: Verification Methodology

409 unbound stages were processed by 5 parallel verification workers.
Each worker:

1. Read the CDG stage description, concept_type, and I/O specs
2. Read the top-3 retrieval candidates' atom descriptions and CDG metadata
3. For ambiguous cases, read the atom's source CDG JSON for full context
4. Judged each match as correct / partial / false_positive

Judgment criteria:
- **correct**: Atom implements the same algorithm. Could be used directly.
- **partial**: Atom implements a component or closely related operation.
  Needs wrapping or combining with other atoms.
- **false_positive**: Shared keywords but different domain/purpose/algorithm.
  Cannot help implement this stage.

Results stored at `/tmp/verify_results_{0-4}.json`.
