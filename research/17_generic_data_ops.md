# Plan: Addressing Generic Data Operation Gaps

## Problem

34 CDG stages describe "generic" data operations (concatenation, loading,
reshaping, math transforms) that are too broad for keyword retrieval to
match reliably. These are not algorithmically novel — they are plumbing —
but they still need resolution for full CDG groundability.

## Why keyword retrieval fails here

These stages use ultra-generic vocabulary ("concatenate", "stack", "merge",
"load", "reshape") that overlaps with hundreds of atoms. The TF-IDF scoring
correctly assigns low weight to these terms, which means the right atom
(if it exists) doesn't surface above noise. The issue isn't missing atoms —
it's that these operations are too generic to discriminate via keywords.

## Proposed resolution strategy

These stages should NOT become atoms. Instead, they should be reclassified
in the CDG bindings using existing resolution mechanisms:

### Category 1: Feature concatenation (14 stages) → `orchestration`

These stages concatenate features from upstream stages. This is pipeline
wiring, not an algorithm.

| CDG | Stage | What it does |
|-----|-------|-------------|
| alaska2_steganalysis_1st | feature_stacking | Concat RGB + YCbCr channels |
| commonlit_readability_4th | concatenation | Merge ranking scores + linguistic features |
| dstl_satellite_features_1st | multi_resolution_concatenation | Upsample + concat multi-band imagery |
| covid_vaccine_mrna_1st | feature_fusion | Concat RNN hidden + GCN embeddings |
| cafa5_protein_function_1st | modality_4_concatenation | Concat sequence + text + literature embeddings |
| eedi_misconception_1st | feature_concatenation | Concat question + answer + misconception embeddings |
| g_research_crypto_1st | data_alignment | Join 14 crypto assets by timestamp |
| handm_personalized_fashion_recommendations_1st | candidate_merging | Union trend + ALS + history candidates |
| ieee_cis_fraud_1st | feature_aggregation | Rolling mean/std of transaction amounts |
| novozymes_enzyme_stability_1st | feature_concatenation | Concat sequence + structure + binding features |
| rsna_mammography_breast_cancer_1st | multi_view_grouping | Group L/R breast views per patient |
| santander_transaction_1st | feature_stacking | Stack binary + count + numeric features |
| trends_neuroimaging_1st | feature_concatenation | Concat fMRI + sMRI features |
| m5_accuracy_1st | temporal_unrolling | Expand aggregated predictions to timesteps |

**Resolution**: Mark as `action_class: "orchestration"` in bindings.
The architect handles concatenation/joining as pipeline wiring between
stages, not as an atomic operation. Where the partial match
`stacking_meta_feature_matrix` was identified, note it as the closest atom
but classify as orchestration.

**Exception**: If the concatenation involves non-trivial logic (upsampling,
temporal alignment, time-decay weighting), those sub-operations should bind
to existing atoms:
- `dstl_satellite_features_1st/multi_resolution_concatenation` → the
  upsample step could bind to `resample_volume` or `scipy.ndimage.zoom`
- `ieee_cis_fraud_1st/feature_aggregation` → already matches
  `rolling_window_features` (partial)
- `m5_accuracy_1st/temporal_unrolling` → `temporal_unroll` or `np.repeat`

### Category 2: Data loading/parsing (9 stages) → `external_knowledge`

These stages load data from specific file formats. File I/O is explicitly
outside the atom boundary (atoms are pure functions, no file I/O).

| CDG | Stage | Format |
|-----|-------|--------|
| cdiscount_image_classification_1st | bson_chunking | BSON |
| child_mind_sleep_states_1st | time_series_unrolling | Parquet |
| byu_flagellar_motors_4th | data_ingestion | cryo-ET + supplements |
| numenta_anomaly_benchmark | streaming_ingestion | CSV/Parquet |
| nasa_pushback_phase1_1st | data_ingestion | Air traffic CSVs |
| google_decimeter_1st | raw_gnss_processing | Android GNSS logs |
| passenger_screening_dhs_1st | format_conversion | .a3d mmWave |
| icecube_neutrinos_1st | point_cloud_processing | IceCube HDF5 |
| open_problems_multimodal_single_cell_1st | sparse_loading | Sparse h5ad |

**Resolution**: Mark as `action_class: "external_knowledge"` in bindings.
Document the file format and parsing library (pydicom, h5py, polars, etc.)
as metadata. The architect treats these as external integration points that
the user provides at runtime.

### Category 3: Array reshaping/formatting (6 stages) → `trivial_inline`

These are trivial numpy operations (reshape, transpose, dtype cast) that
don't warrant a dedicated atom.

| CDG | Stage | Operation |
|-----|-------|-----------|
| amex_default_1st | memory_optimization | float64→float16 downcast |
| openvaccine_mrna_degradation_1st | tensor_formatting | Stack 1D + 2D → 3D tensor |
| seti_breakthrough_listen_1st | array_formatting | Reshape to 6-channel 2D |
| rsna_pe_1st | dicom_windowing | Window + uint8 conversion |
| march_mania_benchmark | season_aggregation | Group-by season average |
| nfl_big_data_bowl_1st | spatial_normalization | Coord flip to common orientation |

**Resolution**: Mark as `action_class: "trivial_inline"` in bindings.
These are 1-3 lines of numpy (np.reshape, .astype, np.stack) and should
be inlined by the architect during code generation, not wrapped as atoms.

**Exception**: `rsna_pe_1st/dicom_windowing` should bind to the new
`dicom_window` atom (partial match was identified).

### Category 4: Simple math transforms (4 stages) → bind to existing atoms

These have existing atoms that should match but were missed:

| CDG | Stage | Should bind to |
|-----|-------|---------------|
| amex_default_1st | difference_features | `temporal_difference` or `create_lag_features` |
| champs_molecular_properties_1st | target_scaling | `standard_scaler_fit` + `standard_scaler_transform` |
| jane_street_market_prediction_1st | streaming_imputation | `forward_fill` |
| moa_prediction_1st | label_smoothing | `label_smoothing_ce` |

**Resolution**: Update bindings to point to the correct existing atoms.
The retrieval missed these due to vocabulary mismatch.

### Category 5: Entity splitting (1 stage) → `orchestration`

| CDG | Stage | Operation |
|-----|-------|-----------|
| nasa_pushback_phase1_1st | entity_splitting | Split by airport ID |

**Resolution**: Mark as `action_class: "orchestration"`. Splitting data by
entity is a MAP_OVER pattern, not an algorithm.

## Summary

| Category | Stages | Resolution | New atoms needed |
|----------|--------|------------|-----------------|
| Feature concatenation | 14 | `orchestration` | 0 |
| Data loading/parsing | 9 | `external_knowledge` | 0 |
| Array reshaping | 6 | `trivial_inline` | 0 |
| Simple math transforms | 4 | Bind to existing atoms | 0 |
| Entity splitting | 1 | `orchestration` | 0 |
| **Total** | **34** | | **0** |

**Key insight**: None of these 34 stages need new atoms. They need
reclassification in the bindings files to reflect that they are pipeline
plumbing, I/O, or trivial operations — not algorithmic atoms.

## Implementation

Update the `_bindings.json` files for each affected CDG:
1. Set `action_class` to the appropriate category
2. Set `status` to "gap" with a descriptive `gap_reason`
3. For math transforms, set `bound_artifact_fqdn` to the matching atom

This closes 34 of the 180 remaining false positives, bringing the true
remaining gap to ~146 stages.
