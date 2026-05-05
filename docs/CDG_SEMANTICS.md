# CDG Semantic Retrieval — What We've Learned

**Scope**: Matching natural-language problem descriptions to solution CDG
templates and evaluating whether the matched template covers the winning
solution's techniques.

**Validation corpus**: 307 new Kaggle competitions (compiled from 6 deep
research batches) + 125 existing CDG templates. 432 total.

---

## The Problem

A user describes a problem in natural language. The architect needs to:
1. Find the best-matching CDG template from the 125 solution templates
2. Determine if the template covers the winning solution's key techniques
3. Ground every stage to an atom

Step 3 is solved — 100% grounding across all 125 CDGs. Steps 1 and 2 are
the retrieval challenge.

---

## What We Tried — Progressive Improvement

### Phase 1: Raw Keyword Overlap

Match the user's prompt against CDG template metadata (`summary`,
`use_when`, `family`, `paradigm`) using raw token intersection counting.

**Result**: 6 competitive (2%), 52 partial (17%), 249 divergent (81%)

**What failed**: Every shared token contributes equally. Generic words
("model", "train", "predict") dominate over discriminative terms.
"EfficientNetV2-S" doesn't match "efficientnet_backbone".

### Phase 2: TF-IDF Scoring Redesign

Replaced raw intersection with TF-IDF weighted overlap. Rare terms
like "decorrelation" or "helix" score 3-5x more than "compute" or "data".
Added extended stop words (60+), port name matching, and category
proximity groups.

**Result for atom retrieval**: Test suite went from 80/83 to 82/83 pass.
Recall@5 hit 100%. This was transformative for atom-level retrieval.

**Result for template retrieval**: Modest improvement. Templates have
longer, more complex descriptions than atoms, so TF-IDF helps less.

### Phase 3: Heuristic Dejargonization

Normalize ML-specific vocabulary before keyword matching. A 130-entry
synonym table maps specific names to canonical categories:
- "EfficientNetV2-S" → "cnn image backbone"
- "CutMix" → "image augmentation region mixing"
- "LightGBM" → "gradient boosting"
- "5-fold StratifiedGroupKFold" → "stratified group cross validation"

Applied to the inbound prompt before keyword search.

**Result**: 8 competitive (3%), 66 partial (21%), 233 divergent (76%)

**What worked**: Bridged vocabulary gaps for well-known terms. Partial
matches jumped from 52 to 66.

**What didn't**: Dejargonization is a lossy mapping — "EfficientNet-B4"
and "ResNet-50" both become "cnn image backbone", losing discriminative
signal. And many competition-specific terms aren't in the synonym table.

### Phase 4: LLM Reranking

After keyword search returns top-N candidates, use an LLM (via Claude CLI
with OAuth session) to semantically evaluate each candidate's fit:
- Problem type match (classification vs regression vs detection)
- Data modality match (tabular vs image vs text)
- Challenge similarity (class imbalance, noisy labels, domain shift)
- Critical stage applicability
- Contraindications (do_not_use_when disqualifiers)

The LLM outputs a JSON ranking with confidence scores and reasoning,
plus a `should_compose_novel` flag when no template fits.

**Result**: 8 competitive (3%), 70 partial (23%), 229 divergent (75%)

**What worked**: LLM correctly reranks templates by semantic problem-type
similarity. Key example: UBC ovarian cancer was reranked from
`hubmap_kidney_1st` (keyword top, score 25.9) to `panda_prostate_mil_1st`
(keyword rank 4, but LLM score 0.87) — MIL on histopathology is a much
better structural match than U-Net segmentation.

**What didn't**: Competitive rate stayed at 8 because technique coverage
was still measured by keyword overlap. The LLM picked better templates
but the scoring didn't reflect it.

### Phase 5: Semantic Technique Coverage

Extended the LLM reranker to also evaluate technique coverage semantically.
For each of the winning solution's `key_techniques`, the LLM determines
whether the matched template has a stage that implements it — using semantic
understanding rather than keyword matching.

"EfficientNet-B0 backbone" is recognized as covered by a stage with
`efficientnet_backbone` atom. "5-fold stratified CV" matches
`cross_validation` stage.

**Result**: 32 competitive (10%), 51 partial (17%), 224 divergent (73%)

**What worked**: Competitive rate jumped 4x (8 → 32). 14 competitions
hit 100% technique coverage. Strong cross-domain matches emerged:
Prudential Insurance → PetFinder (ordinal classification with tabular
features), CityLearn → Lux AI (RL game agent), Recruit Restaurant →
Predict Future Sales (time series forecasting).

**What the 27% competitive+partial rate means**: For roughly 1 in 4
unseen Kaggle competitions, the framework can propose a template that
covers a meaningful portion of the winning solution's techniques.

---

## Key Architectural Insights

### 1. Retrieval is a two-phase problem

Phase 1 (deterministic, cheap): keyword/TF-IDF/embedding retrieval to
get top-N candidates. Must be fast because it runs over all 125 templates.

Phase 2 (LLM, expensive): semantic evaluation of the top-N candidates.
The LLM compares the problem against each template's `applicability`
block (use_when, do_not_use_when, key_insight, failure_modes).

Both phases are essential. Phase 1 alone has 81% divergent. Phase 2
alone is too expensive to run over all 125 templates.

### 2. Dejargonization helps at the margins but isn't transformative

The synonym table bridges known vocabulary gaps but can't generalize
to novel terminology. A competition-specific term like "Macenko stain
normalization" won't be in any synonym table. The LLM handles these
naturally in Phase 2.

Dejargonization is still worth doing because it improves Phase 1 recall
(the keyword search finds more relevant candidates for the LLM to
evaluate) at zero cost.

### 3. Technique coverage must be semantic, not lexical

The single biggest improvement came from LLM-evaluated technique
coverage (Phase 5). Keyword matching between technique names and stage
descriptions has a fundamental vocabulary mismatch — competition winners
describe techniques using specific model names and paper citations,
while CDG templates describe stages using functional descriptions.

Only an LLM can bridge "Swin-v2 with Entmax activation" ↔ "vision
transformer backbone with classification head".

### 4. Template selection and technique scoring are different problems

The LLM reranking (Phase 4) improved template selection quality
significantly but didn't change the competitive rate because technique
scoring was still lexical. Conversely, semantic technique scoring
(Phase 5) changed the competitive rate dramatically.

Both matter, but technique scoring has higher leverage.

### 5. 73% divergent is expected and healthy

Most of the 307 new competitions are genuinely different from our 125
templates. The LLM correctly flags 64% as `should_compose_novel` —
meaning no existing template is a good match and the architect should
compose a new CDG from scratch.

This is the right behavior. The template library covers ~125 problem
types. Kaggle has ~500+ competition types. The remaining 375 need novel
CDGs, which is where the full architect pipeline (decompose → match →
refine) takes over from template retrieval.

### 6. Cross-domain transfer works

The most interesting competitive matches are cross-domain:
- Insurance underwriting → pet adoption (ordinal classification)
- Energy grid optimization → game AI (RL with spatial state)
- Restaurant forecasting → retail sales (time series with holidays)
- Smartphone GPS → smartphone GPS across 4 years (direct transfer)

The CDG template library enables knowledge transfer across competition
domains when the underlying algorithmic structure is similar.

---

## Current Pipeline

```
User prompt
    │
    ▼
┌──────────────────┐
│ Dejargonize      │  130-entry synonym table
│ (heuristic)      │  "EfficientNet" → "cnn image backbone"
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Top-N keyword    │  TF-IDF against dejargonized template
│ retrieval        │  summaries + use_when + stage descriptions
└──────┬───────────┘
       │ top 10 candidates
       ▼
┌──────────────────┐
│ LLM rerank +     │  Semantic evaluation of problem fit +
│ technique        │  per-technique coverage assessment
│ coverage         │  via Claude CLI (OAuth-compatible)
└──────┬───────────┘
       │ best template + coverage report
       ▼
┌──────────────────┐
│ Ground template  │  100% atom grounding across all 125 CDGs
│ stages to atoms  │  2,097 atoms in catalog
└──────────────────┘
```

### Files

| File | Purpose |
|------|---------|
| `sciona/architect/dejargonizer.py` | Heuristic + LLM prompt dejargonization |
| `sciona/architect/solution_index.py` | SolutionTemplateIndex with TF-IDF search |
| `sciona/architect/template_reranker.py` | LLM reranking + semantic technique coverage |
| `sciona/sdk.py` | `find_matching_templates()`, `rerank_templates()`, `propose()` |
| `scripts/validate_kaggle_batch.py` | Batch validation with `--rerank` flag |

---

## What's Next

### Expansion CDGs

Many "partial" matches are close variants of existing templates — same
problem family, different model choices or preprocessing. Expansion CDGs
describe modifications to a base template ("start with melanoma_1st,
swap backbone to ConvNeXt, add tabular metadata branch") without
creating full new CDGs.

### Iterative Refinement

The current validation is single-shot. In practice, the architect
proposes, the user reviews, and the architect refines. Testing this
loop should improve convergence from divergent → partial → competitive.

### Novel CDG Composition

For the 73% divergent competitions, the architect needs to compose CDGs
from scratch using the atom library. This is the full decompose → match
→ synthesize pipeline, not template retrieval. Evaluating this requires
the `Sciona.propose()` path with LLM-backed decomposition.

---

## Numbers at a Glance

| Metric | Value |
|--------|-------|
| CDG templates | 125 |
| Atoms | 2,097 |
| Validation corpus | 307 new competitions |
| CDG grounding | 100% (696/696 stages) |
| Competitive rate (keyword) | 2% (6/307) |
| Competitive rate (semantic) | 10% (32/307) |
| Competitive+Partial (semantic) | 27% (83/307) |
| Template match rate (LLM) | 36% |
| Novel recommended (LLM) | 64% |
| Technique coverage mean | 29.6% |
| Technique coverage >50% | 27% (83/307) |
