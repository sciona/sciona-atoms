# Plan 2: True Remaining Gaps — New Work Needed

**Scope**: 94 stages where no atom exists and reclassification doesn't apply.
**Organized by**: action type (opaque wrapper / symbolic equation / research /
human decision).

---

## A. Opaque Wrappers (20 stages) → Coding Agent

These stages use pretrained models or compiled training frameworks. They
should become `is_opaque=true` atoms defining input/output contracts
without implementing the internals.

**Agent prompt**: For each architecture below, create an opaque atom in
`sciona-atoms-dl/src/sciona/atoms/dl/architectures/` following the pattern
of `efficientnet_backbone` and `resnet_family_backbone`. The atom should
raise `NotImplementedError`, have typed I/O contracts, and document the
expected tensor shapes. Add to `cdg.json` with `is_opaque: true`.

### A1. Pretrained NLP models → `sciona-atoms-dl/architectures/`

| CDG | stage_id | Architecture | Input → Output |
|-----|----------|-------------|---------------|
| bengali_speech_1st | ctc_decoding | CTC decoder head | `(B, T, V) logits → list[str]` |
| cafa5_protein_function_1st | embeddings_generation | ESM-1b protein transformer | `sequence → (D,) embedding` |
| cafa5_protein_function_1st | modality_2_literature | GORetrieval text classifier | `text → (N_classes,) scores` |
| child_mind_sleep_states_1st | transformer_smoothing | Transformer encoder layer | `(B, T, D) → (B, T, D)` |
| commonlit_readability_1st | domain_adaptation | Masked LM pretraining | `tokens → fine-tuned model` |
| eedi_misconception_1st | attention_masking | FlexAttention custom masks | `(B, H, T, T) mask → attention` |
| stanford_ribonanza_1st | graph_transformer_encoder | Graph Transformer | `(N, D) node features → (N, D)` |
| vqa_v2_1st | attention_alignment | Soft attention (text→image) | `(D_text,) + (36, D_img) → (D,)` |
| vqa_v2_1st | feature_fusion | Multimodal bilinear fusion | `(D_text,) + (D_img,) → (D_fused,)` |

### A2. Training pipeline wrappers → `sciona-atoms-dl/training/` or `sciona-atoms-ml/`

| CDG | stage_id | Framework | Contract |
|-----|----------|-----------|---------|
| santander_transaction_1st | lightgbm_training | LightGBM | `(X, y, params) → model` |
| petfinder_adoption_1st | level1_models | XGBoost + LightGBM + TabNet | `(X, y) → predictions` |
| plasticc_1st | model_training | LightGBM multiclass | `(X, y, params) → model` |
| porto_seguro_1st | denoising_autoencoder_features | DAE | `(X) → (X_reconstructed, hidden)` |
| commonlit_readability_1st | stochastic_weight_averaging | SWA optimizer | `(model, lr) → averaged_model` |
| rsna_cervical_spine_fast91_1st | multi_task_heads | Multi-head classifier | `(B, D) → list[(B, C_i)]` |
| m5_uncertainty_1st | base_forecasting | LightGBM + Tweedie | `(X, params) → predictions` |

### A3. Vision models

| CDG | stage_id | Architecture |
|-----|----------|-------------|
| dfdc_deepfake_detection | face_detection | MTCNN / RetinaFace |
| dfdc_deepfake_detection | face_alignment_cropping | Facial landmark alignment |
| icecube_neutrinos_1st | graph_neural_network | DynEdge GNN |
| covid_vaccine_mrna_1st | graph_neural_network | GCN on RNA adjacency |

---

## B. Symbolic Equations (12 stages) → `@symbolic_atom`

These stages implement known mathematical formulas that are ideal
candidates for the SymPy IR system. Each needs a `SymbolicExpression`
with dimensional signatures, validity bounds, and bibliography.

**Agent prompt**: For each equation below, create a `@symbolic_atom` in the
appropriate repo following the pattern in
`sciona-atoms-physics/src/sciona/atoms/physics/particle_tracking/helix_geometry/`.
Each atom needs: `expressions.py` (SymPy Expr), `dimensions.py` (SI dims),
`atoms.py` (@symbolic_atom decorated), `witnesses.py`.

| CDG | stage_id | Equation | Repo |
|-----|----------|----------|------|
| march_mania_benchmark | elo_ratings | Elo rating update: `R_new = R_old + K * (S - E)` where `E = 1/(1 + 10^((R_b - R_a)/400))` | sciona-atoms |
| plasticc_1st | flux_correction | Astronomical flux correction: `F_corrected = F_obs * (1+z)` (K-correction) | sciona-atoms-physics |
| jpx_stock_prediction_1st | target_normalization | Alpha isolation: `alpha = r_stock - r_market` | sciona-atoms-fintech |
| numenta_anomaly_benchmark | windowed_evaluation | NAB scoring: weighted sigmoid window with early-detection reward | sciona-atoms |
| instacart_basket_1st | f1_maximization_dp | Faron's expected F1 maximization: `E[F1] = max_k sum_i P(y_i >= k-th) * 2k / (k + n)` | sciona-atoms |
| santa_2021_magic_minves_1st | problem_reduction | ATSP reduction from string permutation | sciona-atoms-cs |
| santa_2021_magic_minves_1st | distance_matrix_generation | String overlap distance matrix | sciona-atoms-cs |
| commonlit_readability_4th | linguistic_feature_extraction | Flesch-Kincaid / SMOG readability formulas | sciona-atoms-ml |
| m5_uncertainty_1st | hierarchical_scaling | Hierarchical reconciliation (top-down proportional) | sciona-atoms |
| nasa_airport_pushback_phase_1_3rd_place_1st | probabilistic_adjustment | Probabilistic bin adjustment | sciona-atoms |
| osic_pulmonary_1st | confidence_interval_postprocessing | Quantile spread → confidence interval | sciona-atoms |
| two_sigma_news_stock_1st | future_leak_prevention | Temporal alignment rule: `if news.time > 22:00 then assign_date += 1` | sciona-atoms-fintech |

---

## C. Research Needed (7 stages) → Deep Research Prompts

These need another round of deep research to find best-in-class
implementations before atom creation.

### C1. 3D/Volumetric Augmentation

**Research prompt**: Find pure-function implementations for 3D medical image
augmentations: random 3D rotation, elastic deformation, volumetric scaling,
Gaussian noise injection. Source candidates: TorchIO (Apache-2.0),
MONAI (Apache-2.0), or pure scipy.ndimage implementations. Target:
`sciona-atoms-bio` or `sciona-atoms-dl`.

| CDG | stage_id |
|-----|----------|
| byu_flagellar_motors_4th | medical_augmentation |
| vesuvius_ink_detection_1st | volumetric_augmentation |

### C2. Face Detection & Alignment

**Research prompt**: Find pure-function face detection and alignment
implementations. MTCNN has MIT-licensed Python implementations.
RetinaFace has BSD implementations. Need: bounding box extraction from
face detector output + landmark-based alignment crop. Target:
`sciona-atoms-dl`.

| CDG | stage_id |
|-----|----------|
| dfdc_deepfake_detection | face_detection |
| dfdc_deepfake_detection | face_alignment_cropping |

### C3. Chemical Descriptor Extraction

**Research prompt**: Find pure-function implementations for RDKit molecular
descriptor and Morgan fingerprint computation from SMILES strings.
RDKit is BSD-licensed. Need: SMILES → molecular graph → descriptors/fingerprints
as pure functions. Target: `sciona-atoms-bio`.

| CDG | stage_id |
|-----|----------|
| neurips_open_polymer_1st | chemical_descriptor_extraction |

### C4. Morphological Skeletonization

**Research prompt**: Find the pure scipy/skimage implementation of
morphological thinning (skeletonization) for binary masks. skimage has
`skimage.morphology.skeletonize` (BSD). Need: binary mask → 1-pixel-wide
skeleton. Target: `sciona-atoms-dl` or `sciona-atoms`.

| CDG | stage_id |
|-----|----------|
| spacenet3_roads_1st | skeletonization |

### C5. Back-Translation Augmentation

**Research prompt**: Determine whether back-translation (EN→FR→EN) can be
represented as an atom at all, since it requires an external translation
model. Likely conclusion: opaque wrapper or external_tool. If there's a
lightweight implementation using MarianMT, research its interface
contracts. Target: `sciona-atoms-dl`.

| CDG | stage_id |
|-----|----------|
| toxic_comment_1st | back_translation_augmentation |

---

## D. Human Decisions Required (55 stages)

These stages are ambiguous — they could be trivial_inline, orchestration,
approximate bindings, or genuinely new atoms depending on architectural
decisions about scope.

### D1. Feature engineering one-liners (28 stages)

**Decision needed**: Should simple pandas/numpy feature operations (concat,
group-by, ratio computation) be atoms, trivial_inline, or orchestration?

**Recommendation**: Classify as `trivial_inline` or `orchestration`.
These are 1-5 lines of pandas/numpy and don't warrant dedicated atoms.
The architect should inline them during code generation.

Notable exceptions that might warrant atoms:
- `foursquare_location_matching_1st/feature_extraction` — string similarity
  scores (Levenshtein, Jaro-Winkler) are algorithmic, not trivial
- `plasticc_1st/flux_correction` — K-correction is a physics formula → symbolic
- `outbrain_click_prediction_1st/data_leak_extraction` — deterministic leak
  tracing is a graph algorithm

| CDG | stage_id | What it does | Suggested |
|-----|----------|-------------|-----------|
| covid_vaccine_mrna_1st | output_head | Linear layer → predictions | opaque |
| eedi_misconception_1st | prefix_formatting | String formatting | trivial_inline |
| eedi_misconception_1st | feature_extraction | Last-token extraction | trivial_inline |
| foursquare_location_matching_1st | feature_extraction | String similarity scores | **needs atom** (Levenshtein/JW) |
| icecube_neutrinos_1st | point_cloud_processing | Format hits as (x,y,z,t,q) | trivial_inline |
| instacart_basket_1st | feature_merging | Join feature tables | orchestration |
| jane_street_market_prediction_1st | action_space_mapping | Binary target thresholding | trivial_inline |
| jigsaw_toxicity_bias_1st | sequence_bucketing | Length-based batching | orchestration |
| llm_prompt_recovery_1st | embedding_extraction | sentence-transformer inference | opaque |
| lux_ai_season1_1st | state_spatialization | Grid → 3D tensor | trivial_inline |
| m5_uncertainty_1st | hierarchical_scaling | Top-down reconciliation | **symbolic** |
| march_mania_benchmark | pairwise_concatenation | Concat team features | orchestration |
| nasa_pushback_phase1_1st | concatenation | Concat predictions | orchestration |
| ogb_mag240m_1st | graph_sampling | Heterogeneous neighbor sampling | opaque (DGL) |
| osic_pulmonary_1st | feature_concatenation | Concat tabular + image features | orchestration |
| outbrain_click_prediction_1st | data_leak_extraction | Deterministic ID tracing | **needs atom** |
| plasticc_1st | flux_correction | Redshift K-correction | **symbolic** |
| plasticc_1st | feature_merging | Concat GP + metadata features | orchestration |
| porto_seguro_1st | feature_concatenation | Concat tabular + DAE features | orchestration |
| santa_2021_magic_minves_1st | distance_matrix_generation | String overlap distances | **symbolic** |
| santander_transaction_1st | fake_data_identification | Unique-value detection | trivial_inline |
| santander_transaction_1st | real_data_filtering | Filter synthetic rows | trivial_inline |
| santander_transaction_1st | shuffle_augmentation | Column shuffle by class | trivial_inline |
| sartorius_cell_segmentation_1st | crop_extraction | ROI crop from bbox | trivial_inline |
| shopee_price_match_1st | dynamic_thresholding | Score → binary adjacency | trivial_inline |
| spacenet3_roads_1st | multispectral_stacking | Stack GeoTIFF bands | trivial_inline |
| um_mcts_strength_1st | target_smoothing | Target recomputation | trivial_inline |
| vqa_v2_1st | bottom_up_image_extraction | Faster R-CNN ROI extraction | opaque |

### D2. Post-processing rules (9 stages)

**Decision needed**: Are competition-specific post-processing rules worth
atomizing? They're deterministic but highly specific.

**Recommendation**: Most should be `trivial_inline`. The wavelet denoising
is the exception — it's a real algorithm.

| CDG | stage_id | Suggested |
|-----|----------|-----------|
| feedback_prize_writing_1st | length_constraint_filtering | trivial_inline |
| great_barrier_reef_1st | boundary_filtering | trivial_inline |
| instacart_basket_1st | candidate_generation | trivial_inline (set filter) |
| otto_recommender_1st | post_processing | trivial_inline (sort + truncate) |
| panda_prostate_mil_1st | white_space_culling | trivial_inline (threshold) |
| rsna_pe_1st | post_processing_logic | trivial_inline (rule-based) |
| two_sigma_news_stock_1st | future_leak_prevention | **symbolic** (temporal alignment) |
| um_mcts_strength_1st | 8_0_2_weighting_schema_numpy | trivial_inline |
| vsb_power_line_1st | wavelet_denoising | **needs research** (PyWavelets DWT) |

### D3. Genuinely ambiguous (16 stages)

These need case-by-case human judgment:

| CDG | stage_id | Question |
|-----|----------|----------|
| byu_flagellar_motors_4th | medical_augmentation | Research (3D augmentation) or opaque (MONAI)? |
| commonlit_readability_1st | stochastic_weight_averaging | Is SWA an atom or a training infrastructure concern? |
| eedi_misconception_1st | attention_masking | Is FlexAttention an atom or architecture-specific? |
| facebook_image_similarity_1st | spatial_verification | Research needed: RANSAC + homography from keypoints |
| feedback_prize_writing_1st | char_to_token_offset_mapping | We have char_to_token_offsets — is it a retrieval miss or different algo? |
| hpa_single_cell_classification_1st | label_assignment | argmax label assignment — trivial_inline? |
| jpx_stock_prediction_1st | target_normalization | Symbolic equation (alpha = r_stock - r_market)? |
| lanl_earthquake_1st | stacking | Ensemble averaging with physical bounds — fold_ensemble_average partial? |
| llm_science_exam_1st | context_concatenation | String formatting — trivial_inline? |
| llm_science_exam_1st | confidence_weighting | Prediction averaging — voting_regressor_average? |
| passenger_screening_dhs_1st | body_zoning | Domain-specific (17 anatomical zones from 2D projection) |
| stanford_ribonanza_1st | dual_head_output | Multi-head output + L1 loss — opaque + loss atom? |
| toxic_comment_1st | back_translation_augmentation | Research needed: MarianMT wrapper or external_tool? |
| trends_neuroimaging_1st | brain_mask_application | 3D voxel mapping — nibabel I/O or spatial atom? |
| vsb_power_line_1st | signal_to_matrix_transform | Reshape 1D → 2D — trivial_inline (np.reshape)? |
| eedi_misconception_1st | data_deduplication | pandas drop_duplicates — trivial_inline? |

---

## Summary

| Action type | Stages | Who executes |
|-------------|--------|-------------|
| **Opaque wrappers** | 20 | Coding agent — define contracts only |
| **Symbolic equations** | 12 | Coding agent with SymPy — `@symbolic_atom` |
| **Deep research** | 7 | Research prompt → deep research → ingestion agent |
| **Human: feature eng** | 28 | Human classifies, agent reclassifies bindings |
| **Human: postprocessing** | 9 | Human classifies, agent reclassifies bindings |
| **Human: ambiguous** | 16 | Human decides per-stage |
| **Already in Plan 1** | 2 | (wavelet denoising → research, flux → symbolic) |
| **Total** | **94** | |

### Recommended priority order

1. **Opaque wrappers** (20) — high leverage, low effort, unblocks architecture stages
2. **Symbolic equations** (12) — showcase the SymPy system, well-defined formulas
3. **Human decisions on feature eng** (28) — most are trivial_inline, batch classify
4. **Human decisions on postprocessing** (9) — same pattern
5. **Deep research** (7) — 3D augmentation, face detection, SMILES, skeletonization
6. **Genuinely ambiguous** (16) — case-by-case
