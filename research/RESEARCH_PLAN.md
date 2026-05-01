# Atom Research Plan — RC1 Gap Coverage

**Created**: 2026-04-25
**Goal**: Find best-in-class implementations for ~317 missing atoms across 125 CDGs
**Method**: Deep research organized by source library/domain, yielding implementation
specifications that ingestion agents can turn into atoms

## How to use this plan

1. Pick a research project below
2. Run deep research using the corresponding prompt file (`research/NN_*.md`)
3. Save the research output alongside the prompt
4. Hand the output + `AGENT_INGESTION.md` to an ingestion agent for atom creation

Each research project is independent and can run in parallel.

## Research Projects

| # | File | Domain | Expected atoms | Target repo | Priority |
|---|------|--------|---------------|-------------|----------|
| 01 | `01_image_augmentation.md` | Image augmentation & TTA | ~25 | sciona-atoms-dl | P0 |
| 02 | `02_loss_functions.md` | Specialized loss functions | ~15 | sciona-atoms-dl | P0 |
| 03 | `03_detection_postprocessing.md` | Object detection post-processing | ~15 | sciona-atoms-dl | P0 |
| 04 | `04_text_nlp_encoding.md` | Text processing & NLP encoding | ~20 | sciona-atoms-ml + dl | P0 |
| 05 | `05_audio_speech.md` | Audio & speech processing | ~15 | sciona-atoms-signal | P1 |
| 06 | `06_medical_imaging_io.md` | Medical imaging I/O & preprocessing | ~15 | sciona-atoms-bio | P1 |
| 07 | `07_cnn_architectures.md` | CNN/architecture opaque wrappers | ~20 | sciona-atoms-dl | P1 |
| 08 | `08_video_temporal.md` | Video frame extraction & temporal ops | ~12 | sciona-atoms-dl | P1 |
| 09 | `09_graph_construction.md` | Graph construction & GNN utilities | ~15 | sciona-atoms + dl | P1 |
| 10 | `10_gradient_boosting_tabular.md` | Gradient boosting & tabular ML | ~12 | sciona-atoms-ml | P1 |
| 11 | `11_embeddings_retrieval.md` | Embedding extraction & similarity search | ~15 | sciona-atoms-dl | P2 |
| 12 | `12_segmentation_morphology.md` | Segmentation & morphological ops | ~12 | sciona-atoms-dl | P2 |
| 13 | `13_recommender_systems.md` | Recommender system primitives | ~10 | sciona-atoms-dl | P2 |
| 14 | `14_time_series_features.md` | Time series feature engineering | ~10 | sciona-atoms-signal | P2 |
| 15 | `15_geospatial_sensors.md` | Geospatial, GNSS, sensor fusion | ~10 | sciona-atoms-geo | P2 |
| 16 | `16_bayesian_bandits.md` | Bayesian optimization & bandits | ~8 | sciona-atoms | P2 |

**Total estimated atoms: ~219**

## Priority definitions

- **P0**: Highest leverage — covers the most CDG stages per atom, well-defined implementations
- **P1**: Medium leverage — important domains but more specialized
- **P2**: Lower leverage — fewer CDGs affected or more domain-specific

## What the research output should contain

For each candidate atom, the research should provide:

1. **Canonical implementation source** — URL to the best reference implementation
2. **License** — must be compatible (MIT, BSD, Apache-2.0, or public domain)
3. **Pure function boundary** — where to draw the atom's function signature
   (inputs, outputs, no side effects, no GPU state, no file I/O)
4. **Contract opportunities** — natural preconditions and postconditions
   (e.g., "input array must be 2D", "output sums to 1.0")
5. **Witness template** — what a minimal test case looks like
6. **CDG stages covered** — which specific CDG stages this atom would bind to
7. **Concept type** — which ConceptType enum value applies
8. **Dependencies** — what libraries are needed (numpy-only preferred,
   torch/scipy acceptable, heavy deps like FAISS noted)

## Ingestion workflow (for the agent receiving research output)

1. Read `AGENT_INGESTION.md` for the full ingestion protocol
2. For each atom in the research output:
   a. Write `atoms.py` with `@register_atom`, typed signature, contracts
   b. Write `witnesses.py` with concrete test cases
   c. Write/update `cdg.json` with the atom node
   d. Run verification: `run_mypy`, `validate_cdg_ir`, `run_contribution_check`
3. Create bindings in the corresponding solution CDG `_bindings.json`
