# Research: Kaggle Competition Corpus

## CRITICAL: Output format

**You MUST output a single JSON array. No essays. No narrative. No meta-analysis.**

Your entire output should be a JSON array of competition objects. Nothing else.
If you want to add notes, put them in the `"notes"` field of individual entries.

## What to produce

A JSON array of 50 competition objects. Each object has this EXACT schema:

```json
{
  "competition_id": "google-smartphone-decimeter-2021",
  "title": "Google Smartphone Decimeter Challenge",
  "year": 2021,
  "problem_type": "regression",
  "domain": "geospatial",
  "modalities": ["tabular", "time_series"],
  "evaluation_metric": "50th percentile horizontal distance error (meters)",
  "prompt": "Predict precise smartphone GPS location from raw GNSS satellite measurements and IMU sensor data. Data: pseudorange observations at 1Hz from multiple satellite constellations, accelerometer/gyroscope at 100Hz, ~500 driving traces of 5-30 minutes each. Ground truth from NovAtel reference system. Metric: median horizontal distance error in meters. Constraints: no external correction services at inference time.",
  "solution_summary": "EKF sensor fusion with RTS backward smoother. Filtered satellites by C/N0 > 25 dB-Hz. Applied phone-specific clock bias correction. Snapped final trajectory to road network. Key insight: phone-specific systematic biases dominate over GNSS accuracy.",
  "key_techniques": [
    "Extended Kalman Filter for GNSS+IMU fusion",
    "Rauch-Tung-Striebel backward smoother",
    "C/N0 satellite quality filtering",
    "Road network snap-to-grid post-processing",
    "Phone-specific bias correction"
  ],
  "source_url": "https://www.kaggle.com/competitions/google-smartphone-decimeter-challenge/discussion/..."
}
```

## Field definitions

- `competition_id`: Kaggle URL slug exactly as it appears in kaggle.com/competitions/SLUG
- `title`: Full competition title
- `year`: Year competition ended
- `problem_type`: One of: "classification", "regression", "detection", "segmentation", "ranking", "recommendation", "optimization", "generation", "matching"
- `domain`: One of: "tabular", "cv", "nlp", "audio", "medical", "geospatial", "time_series", "graph", "recommender", "scientific", "multimodal", "rl"
- `modalities`: List from: "tabular", "image", "text", "audio", "video", "3d_volume", "graph", "time_series", "geospatial", "code", "molecular"
- `evaluation_metric`: Exact metric name and brief formula if non-standard
- `prompt`: 2-4 sentences describing the problem AS A USER WOULD STATE IT. This is NOT the competition page text. It should say: what to predict, what data is available, what the metric is, and what the key challenges are. Write it as if you're asking an AI assistant for help.
- `solution_summary`: 2-4 sentences covering: overall pipeline, key model choices, critical preprocessing, ensemble strategy, decisive insight. Be SPECIFIC — name exact models, techniques, and parameters.
- `key_techniques`: List of 4-8 SPECIFIC techniques. Say "EfficientNet-B4" not "CNN". Say "5-fold stratified GroupKFold" not "cross-validation". Say "CutMix + MixUp augmentation" not "data augmentation".
- `source_url`: URL to the 1st-place write-up (Kaggle discussion, GitHub, blog)

## Which competitions to include

Focus on Featured and Research competitions from 2018-2026 with publicly available 1st-place solution write-ups. Start with the most well-known and high-prize competitions.

Good sources for finding these:
- https://farid.one/kaggle-solutions/ (comprehensive list with links)
- Kaggle competition discussion forums (search "1st place solution")
- https://github.com/interviewBubble/Kaggle-Solutions

## Competitions to SKIP

- We already have CDGs for these 125 competitions. Do NOT include any of them:
  adversarial_attacks, aimo_llm_tool_use, alaska2_steganalysis, alice_lyric_alignment,
  amex_default, aptos_blindness, arc_program_synthesis, avito_demand,
  barachant_seizure, bengali_grapheme, bengali_speech, birdclef_edge_device_2021,
  bms_molecular_translation, byu_flagellar_motors, cafa5_protein_function,
  cassava_leaf, cause_effect, cdiscount_image_classification, chaii_qa,
  champs_molecular_properties, child_mind_sleep_states, commonlit_readability,
  connectomics, cornell_birdcall, covid_vaccine_mrna, dcase2020_sound_event_detection,
  dfdc_deepfake_detection, dfl_bundesliga, dsb2017, dstl_satellite_features,
  eedi_misconception, facebook_image_similarity, feedback_prize_writing,
  flavours_physics, foursquare_location_matching, g_research_crypto,
  global_wheat, google_asl_fingerspelling, google_asl_translation,
  google_contrails, google_decimeter, google_landmark_retrieval,
  google_universal_image_embedding, great_barrier_reef, halite_two_sigma,
  handm_personalized_fashion, home_credit_default, hpa_single_cell,
  hubmap_hpa_human_body, hubmap_kidney, hubmap_vasculature, icecube_neutrinos,
  ieee_cis_fraud, image_matching, indoor_navigation, instacart_basket,
  jane_street_market_prediction, jigsaw_toxicity_bias, jpx_stock_prediction,
  kaggle_ner, lanl_earthquake, llm_prompt_recovery, llm_science_exam,
  lux_ai_season1, lyft_3d_object_detection, lyft_motion, m5_accuracy,
  m5_uncertainty, make_data_count, march_mania, melanoma, mercari_price,
  miccai_tn_scui, moa_prediction, nasa_airport_pushback, nasa_pushback_phase1,
  neurips_open_polymer, nfl_big_data_bowl, nfl_health_and_safety,
  novozymes_enzyme_stability, numenta_anomaly_benchmark, ogb_mag240m,
  ogb_wikikg90m, open_problems_multimodal_single_cell, openvaccine_mrna,
  optiver_realized_volatility, osic_pulmonary, otto_group_product,
  otto_recommender, outbrain_click_prediction, panda_prostate_mil,
  parkinsons_fog, passenger_screening, petfinder_adoption, plasticc,
  playground_s5e4_podcast, playground_s6e3_churn, porto_seguro,
  predict_future_sales, rsna_cervical_spine, rsna_mammography,
  rsna_miccai_brain_tumor, rsna_pe, rsna_pneumonia, santa_2020_candy_cane,
  santa_2021_magic_minves, santander_transaction, sartorius_cell_segmentation,
  seti_breakthrough_listen, severstal_steel, shopee_price_match,
  spacenet3_roads, stanford_ribonanza, tgs_salt_identification,
  toxic_comment, trackml, trends_neuroimaging, two_sigma_news_stock,
  um_mcts_strength, vesuvius_ink_detection, vqa_v2, vsb_power_line,
  web_traffic_forecasting

## REMINDER

Output ONLY the JSON array. Start with `[` and end with `]`. No markdown fences.
No preamble. No conclusion. Just the data.
