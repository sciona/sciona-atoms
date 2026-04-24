# Solution CDG Authoring Guide

This guide defines the structure and quality bar for solution CDG JSON files
stored in `sciona-atoms/data/solution_cdgs/`. These CDGs serve as
**deterministic pipeline templates** that the architect matches against when
decomposing new problems.

## Two-Phase Retrieval Model

CDGs participate in a two-phase retrieval pipeline:

1. **Phase 1 (Deterministic, cheap):** Embedding similarity, keyword overlap,
   and family/paradigm matching produce a ranked shortlist of 3-5 candidate
   CDGs. This phase reads ONLY the top-level metadata fields — it never
   touches the stage descriptions. Tokens spent: zero (all vector math).

2. **Phase 2 (LLM reasoning, expensive):** The architect reads the shortlisted
   CDGs in full and reasons about which template best fits the new problem.
   This is where the CDG must carry enough structured signal to support
   comparison without the architect needing to read every stage description
   line by line.

**Design principle:** Make Phase 1 filtering effective through rich metadata.
Make Phase 2 comparison efficient through structured applicability fields.

---

## Required Top-Level Fields

### Metadata (used by Phase 1 deterministic matching)

```json
{
  "asset_id": "solution.kaggle.{competition}_{placement}",
  "asset_version": "v1",
  "family": "tabular_classification",
  "paradigm": "optimization",
  "name": "Human-readable name — Competition Placement — Key Technique",
  "summary": "1-2 sentences. Technical. Names the core techniques.",
  "dejargonized_summary": "1-2 sentences. Plain English. A non-expert can follow.",
  "variant_hints": ["keyword1", "keyword2", ...],
  "inputs": [...],
  "outputs": [...]
}
```

**`family`**: The problem type. Use snake_case. Examples:
tabular_classification, tabular_regression, nlp_classification,
nlp_regression, nlp_span_detection, nlp_fairness, object_detection_2d,
semantic_segmentation, multimodal_retrieval, multimodal_classification,
time_series_classification, time_series_forecasting, signal_classification,
survival_regression, set_prediction, trajectory_prediction, sensor_fusion,
graph_inference, structured_prediction, medical_detection_2d,
medical_imaging_3d, gigapixel_segmentation, image_forensics,
geometric_matching, multilabel_tabular, medical_image_tabular,
tabular_text_regression, adversarial, constrained_ml, riemannian_bci,
particle_tracking, combinatorial_optimization.

**`paradigm`**: The dominant algorithmic approach. Use ConceptType enum values:
optimization, neural_network, signal_transform, geometry, graph_signal_processing,
signal_filter, analysis, etc.

**`variant_hints`**: Keywords that help Phase 1 retrieval. Include:
- Technique names (denoising_autoencoder, arcface, pseudo_labeling)
- Data characteristics (highly_imbalanced, irregularly_sampled, multi_label)
- Domain markers (medical, satellite, financial, text, audio)

### Applicability (used by Phase 2 reasoning)

These fields help the architect compare shortlisted CDGs efficiently.

```json
{
  "applicability": {
    "use_when": [
      "Tabular data with >50 features and high class imbalance (>10:1)",
      "Missing values encoded as sentinel values (-1, 999, etc.)",
      "No strong domain-specific feature engineering available"
    ],
    "do_not_use_when": [
      "Data has <10 features (DAE adds no value on small feature spaces)",
      "Target is multi-class with >10 classes (rank-averaging is less effective)",
      "Real-time inference required (DAE feature extraction adds latency)"
    ],
    "key_insight": "The DAE learns a smooth manifold of the feature space. Its hidden activations capture non-linear feature interactions that tree models cannot discover from raw features alone. The swap-noise corruption forces the DAE to learn robust representations rather than memorizing.",
    "critical_stages": ["denoising_autoencoder_features"],
    "swappable_stages": {
      "model_training": "Any supervised model (GBM, NN, linear) works here. The value is in the DAE features, not the downstream model.",
      "rank_average_ensemble": "Any ensembling strategy (stacking, blending, simple average) works. Rank-averaging is just robust to different prediction scales."
    },
    "scaling_notes": "DAE training time scales with O(features × samples × epochs). For >1M rows, use mini-batch training. For >1000 features, consider PCA before DAE.",
    "failure_modes": [
      "Very small datasets (<1000 rows): DAE overfits and produces noisy features",
      "Highly correlated features: DAE hidden layer collapses to first few PCA components",
      "Time-series data: DAE ignores temporal ordering, producing inferior features vs. lag-based approaches"
    ]
  }
}
```

Field definitions:

- **`use_when`**: 2-4 conditions that make this pipeline the right choice.
  Be specific about data properties (size, dimensionality, imbalance ratio,
  modality). The architect compares these across candidates.

- **`do_not_use_when`**: 2-4 conditions where this pipeline fails or is
  suboptimal. Helps the architect eliminate candidates quickly.

- **`key_insight`**: 1-2 sentences explaining WHY this pipeline works — the
  non-obvious mechanism behind its success. This is the single most
  important field for the architect's reasoning.

- **`critical_stages`**: Stage IDs that are essential to the pipeline's
  success. Removing or replacing these would fundamentally change the
  approach. Helps the architect understand what's load-bearing.

- **`swappable_stages`**: Stage IDs with descriptions of what alternatives
  work. Helps the architect customize the template without breaking it.

- **`scaling_notes`**: How the pipeline behaves at different data scales.
  Optional but valuable for production-oriented reasoning.

- **`failure_modes`**: Specific scenarios where the pipeline breaks. Helps
  the architect avoid proposing a template for the wrong problem.

---

## Stage-Level Fields

Each stage in the `stages` array must have:

```json
{
  "stage_id": "snake_case_unique_within_cdg",
  "name": "Human Readable Name",
  "description": "Technical description. Names specific algorithms and parameters.",
  "dejargonized_description": "Plain English. A product manager can follow.",
  "concept_type": "ConceptType enum value",
  "inputs": [{"name": "...", "type_desc": "...", "constraints": "...", "required": true}],
  "outputs": [{"name": "...", "type_desc": "...", "constraints": "...", "required": true}],
  "preconditions": ["..."],
  "guarantees": ["..."],
  "matched_primitive": ""
}
```

**`concept_type`** must be a valid ConceptType enum value. Common values:
signal_filter, signal_transform, neural_network, optimization, data_assembly,
data_extraction, dimensionality_reduction, clustering, analysis, randomized,
loss_function, searching, geometry, conditional_routing, dynamic_programming,
graph_traversal, sequential_filter, external_tool, external_knowledge.

**`matched_primitive`**: Leave empty for new CDGs. The binding process fills
this in later after retrieval testing.

---

## Edge-Level Fields

```json
{
  "source_stage_id": "stage_a",
  "target_stage_id": "stage_b",
  "output_name": "features",
  "input_name": "features",
  "source_type": "NDArray[np.float64]",
  "target_type": "NDArray[np.float64]",
  "data_kind": "feature_vector",
  "provenance": "Brief explanation of what flows through this edge.",
  "loss_class": "preserving"
}
```

For loss functions connected via callable injection:
```json
{
  "edge_kind": "callable_injection",
  "data_kind": "callable"
}
```

---

## Planning Constraints

Include 2-3 constraints that capture ordering requirements or hard rules.
These help the architect avoid invalid stage orderings.

```json
{
  "category": "stage",
  "subject": "imputation_first",
  "statement": "Missing value handling must precede all feature engineering.",
  "rationale": "Downstream stages assume clean numeric inputs.",
  "confidence": 1.0,
  "source_stage": "solution_analysis",
  "source_reference": "Competition writeup URL or analysis doc"
}
```

---

## Audit Block

```json
{
  "audit": {
    "source_kind": "kaggle_solution",
    "review_status": "draft",
    "rationale": "1-2 sentences on why this solution was selected for CDG modeling.",
    "dejargonized_summary": "Plain English version of the rationale.",
    "provenance_notes": ["Source URL", "License", "Local analysis doc path"],
    "uncertainty_notes": ["What we're not sure about"],
    "references": [{"title": "...", "citation": "...", "note": "..."}],
    "maintainers": ["sciona-atoms-ml"]
  }
}
```

---

## Quality Checklist

Before committing a CDG:

- [ ] `family` accurately describes the problem type
- [ ] `variant_hints` include technique names AND data characteristics
- [ ] `applicability.use_when` has 2+ specific, testable conditions
- [ ] `applicability.do_not_use_when` has 2+ specific failure conditions
- [ ] `applicability.key_insight` explains WHY, not just WHAT
- [ ] `applicability.critical_stages` identifies load-bearing stages
- [ ] `applicability.swappable_stages` identifies customizable stages
- [ ] Every stage has a valid `concept_type` from the enum
- [ ] Every stage has typed `inputs` and `outputs`
- [ ] Edges form a valid DAG (no cycles)
- [ ] At least 2 `planning_constraints` capture ordering rules
- [ ] `audit.provenance_notes` includes the source URL
- [ ] JSON is valid (`python -c "import json; json.load(open('file.json'))"`)
