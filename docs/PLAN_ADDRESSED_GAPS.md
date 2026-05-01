# Plan 1: Technically Addressed Gaps — Detail Work

**Scope**: 66 stages where atoms exist or stages should be reclassified.
**Executor**: Coding agents with specific instructions per batch.
**Prerequisite**: None — all information needed is in this document.

---

## Batch A: Alias Fixes (27 stages) → Coding Agent

These stages have atoms found by deep search but missed by retrieval.
The cdg.json aliases have already been added for most. What remains is
updating the `_bindings.json` files to set `status: "active"` and
`bound_artifact_fqdn` to the correct atom FQDN.

**Agent prompt**: For each row below, update the corresponding
`_bindings.json` file in `sciona-atoms/data/solution_cdgs/`. Set
`status: "active"`, `action_class: "replace_stage"`,
`binding_source: "retrieval_search"`, and `bound_artifact_fqdn` to the
FQDN shown.

| CDG | stage_id | Atom FQDN |
|-----|----------|-----------|
| alaska2_steganalysis_1st | spatial_decompression | sciona.atoms.scipy.idct |
| alice_lyric_alignment_1st | text_processing | sciona.atoms.ml.text_nlp.rule_based_g2p |
| bms_molecular_translation_1st | decoding | sciona.atoms.ml.text_nlp.beam_search |
| cassava_leaf_1st | ensembling | sciona.atoms.dl.image_augmentation.tta_geometric_average |
| cdiscount_image_classification_1st | test_time_augmentation_tta | sciona.atoms.dl.image_augmentation.tta_10crop_average |
| commonlit_readability_1st | tokenization | sciona.atoms.ml.tokenizer.tokenize |
| chaii_qa_1st | tokenization | sciona.atoms.ml.tokenizer.tokenize |
| dstl_satellite_features_1st | conditional_random_fields_crf | sciona.atoms.dl.segmentation.dense_crf_2d |
| facebook_image_similarity_1st | database_side_augmentation_query_expansion | sciona.atoms.dl.embeddings.alpha_query_expansion |
| handm_personalized_fashion_recommendations_1st | collaborative_filtering | sciona.atoms.dl.recommender.als_user_update |
| handm_personalized_fashion_recommendations_1st | trend_generation | sciona.atoms.dl.recommender.item_popularity_decay |
| jigsaw_toxicity_bias_1st | text_cleaning | sciona.atoms.ml.text_nlp.clean_text |
| jigsaw_toxicity_bias_1st | multi_tokenization | sciona.atoms.ml.tokenizer.tokenize |
| kaggle_ner_1st | sequence_tokenization | sciona.atoms.ml.tokenizer.tokenize |
| kaggle_ner_1st | decoding | sciona.atoms.ml.text_nlp.bio_decode |
| lanl_earthquake_1st | signal_chunking | sciona.atoms.dl.video_temporal.sliding_windows |
| melanoma_1st | microscope_augmentation | sciona.atoms.dl.image_augmentation.cutmix_apply |
| moa_prediction_1st | variance_thresholding | sciona.atoms.ml.sklearn.variance_threshold.variance_threshold_fit |
| nasa_airport_pushback_phase_1_3rd_place_1st | feature_engineering | sciona.atoms.signal.time_series_features.rolling_window_features |
| nfl_health_and_safety_helmet_assignment_1st | object_detection | sciona.atoms.dl.architectures.yolo_object_detector |
| optiver_realized_volatility_1st | cross_sectional_aggregation | sciona.atoms.ml.tabular.gradient_boosting.group_aggregate |
| optiver_realized_volatility_1st | feature_clustering | sciona.atoms.ml.sklearn.cluster.kmeans_plusplus.kmeans_plusplus_initialize_dense |
| outbrain_click_prediction_1st | feature_hashing | sciona.atoms.ml.sklearn.feature_extraction.feature_hasher_shell.feature_hasher_csr_matrix |
| plasticc_1st | time_series_feature_extraction | sciona.atoms.signal.time_series_features.rolling_window_features |
| predict_future_sales_benchmark | mean_encoding | sciona.atoms.ml.tabular.gradient_boosting.target_encode |
| rsna_pe_1st | dicom_windowing | sciona.atoms.bio.medical_imaging_3d.preprocessing.dicom_window |
| santa_2020_candy_cane_1st | action_execution | sciona.atoms.inference.bayesian_bandits.select_best_arm |

## Batch B: Metadata Fixes (15 stages) → Coding Agent

These atoms partially match but have description or concept_type issues
preventing reliable retrieval. Fix the atom's cdg.json and/or the
bindings.

**Agent prompt**: For each row, update the binding to `status: "approximate"`
with the atom FQDN and a note about what adaptation is needed.

| CDG | stage_id | Partial atom | Fix needed |
|-----|----------|-------------|------------|
| chaii_qa_1st | post_processing_logic | char_to_token_offsets | Bind as approximate; needs span selection logic on top |
| g_research_crypto_1st | cross_asset_features | pairwise_ratios | Bind as approximate; generic ratios, not BTC-specific |
| global_wheat_1st | mosaic_augmentation | cutmix_apply | Bind cutmix as partial; mosaic (4-image) still missing |
| google_decimeter_1st | raw_gnss_processing | correct_clock_bias | Bind as approximate; covers corrections but not raw log parsing |
| great_barrier_reef_1st | data_augmentation | mixup_apply | Bind as partial; bbox-safe augmentation still missing |
| hubmap_vasculature_1st | mask_prediction | unet_2d_segmentation | Bind as opaque architecture |
| instacart_basket_1st | user_product_aggregation | group_aggregate | Bind as approximate; covers aggregation, not full interaction features |
| llm_science_exam_1st | offline_indexing | build_faiss_flat_ip | Bind as approximate; add "dense vector indexing" to atom description |
| make_data_count_finding_data_references_2nd_place_1st | regex_filtering | clean_text | Bind as approximate; covers text cleaning, not regex pattern matching |
| moa_prediction_1st | deep_tabular_models | entity_embedding_lookup | Bind embedding layer as approximate; full TabNet is opaque |
| openvaccine_mrna_degradation_1st | gcn_layer | adjacency_smoothing | Bind as approximate; update description to mention GCN |
| osic_pulmonary_1st | tabular_feature_engineering | temporal_difference | Bind as approximate; covers deltas, not full feature set |
| santander_transaction_1st | count_encoding | frequency_encode | Bind as active (exact match, just vocabulary mismatch) |
| santander_transaction_1st | value_substitution | frequency_encode | Bind as active |
| toxic_comment_1st | tokenization_and_embedding | tokenize | Bind tokenize as partial; embedding layer separate |

## Batch C: Reclassify as external_knowledge (18 stages) → Coding Agent

**Agent prompt**: For each row, update the binding to
`action_class: "external_knowledge"`, `status: "gap"` with a
`gap_reason` noting the file format or external system.

| CDG | stage_id | Reason |
|-----|----------|--------|
| byu_flagellar_motors_4th | data_ingestion | cryo-ET file loading |
| cdiscount_image_classification_1st | bson_chunking | BSON format parsing |
| cdiscount_image_classification_1st | multi_gpu_data_loader | PyTorch DistributedDataParallel infrastructure |
| child_mind_sleep_states_1st | time_series_unrolling | Parquet streaming ingestion |
| google_decimeter_1st | raw_gnss_processing | Android GNSS log parsing (partial — corrections are atoms) |
| indoor_navigation_1st | wifi_feature_extraction | WiFi RSSI signal database lookup |
| indoor_navigation_1st | wifi_position_model | WiFi fingerprint model (external database) |
| numenta_anomaly_benchmark | streaming_ingestion | CSV streaming reader |
| nasa_pushback_phase1_1st | data_ingestion | Air traffic CSV loading |
| ogb_mag240m_1st | node_text_extraction | OGB dataset API loading |
| open_problems_multimodal_single_cell_1st | sparse_loading | Sparse h5ad matrix loading |
| openvaccine_mrna_degradation_1st | base_pair_extraction | ViennaRNA tool invocation |
| passenger_screening_dhs_1st | format_conversion | Proprietary .a3d format |
| playground_s5e4_podcast_1st | automated_data_ingestion | AutoGluon data loading |
| rsna_mammography_breast_cancer_1st | multi_view_grouping | DICOM series grouping by patient |
| rsna_pneumonia_1st | dicom_extraction | DICOM to PNG conversion |
| stanford_ribonanza_1st | base_pair_probability_bpp_extraction | EternaFold tool invocation |
| covid_vaccine_mrna_1st | secondary_structure_parsing | RNA folding tool (Arnie/ViennaRNA) |

## Batch D: Reclassify as external_tool / orchestration / trivial (6 stages) → Coding Agent

| CDG | stage_id | New action_class | Reason |
|-----|----------|-----------------|--------|
| aimo_llm_tool_use_topology | prompt_injection | external_tool | LLM prompt engineering |
| aimo_llm_tool_use_topology | code_generation | external_tool | LLM code generation |
| playground_s5e4_podcast_1st | automated_stacking | external_tool | AutoGluon stacking |
| severstal_steel_1st | multistage_inference | orchestration | Conditional routing pattern |
| predict_future_sales_benchmark | target_clipping | trivial_inline | `np.clip(target, 0, 20)` |
| nfl_big_data_bowl_1st | spatial_tensor_formatting | trivial_inline | `np.sort` + flatten |

---

**Total Plan 1: 66 stages, 0 new atoms needed, 4 coding agent batches.**
