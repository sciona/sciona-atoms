# Plan: Kaggle Competition Validation — End-to-End Framework Audit

**Goal**: Verify that sciona can produce competitive solutions for real Kaggle
problems by feeding competition descriptions into the architect and comparing
the proposed CDG pipelines against documented winning solutions.

**Success metric**: For each competition, the architect's proposed CDG should
cover the critical algorithmic decisions of the winning solution — not
necessarily identical, but demonstrably competitive.

---

## Phase 1: Compile Competition Corpus

### 1.1 Source catalog

Build a structured dataset of Kaggle competitions with available winning
solution write-ups. For each competition, collect:

```json
{
  "competition_id": "google-smartphone-decimeter-2021",
  "title": "Google Smartphone Decimeter Challenge",
  "year": 2021,
  "problem_type": "regression",
  "domain": "geospatial/sensors",
  "metric": "distance_percentile_50",
  "prompt": {
    "description": "Full competition description text",
    "data_description": "What data is provided (features, formats, sizes)",
    "evaluation_metric": "Exact metric formula and scoring rules",
    "constraints": "Time limits, resource limits, submission format"
  },
  "winning_solutions": [
    {
      "placement": "1st",
      "team": "Team name",
      "source_url": "Discussion post / GitHub URL",
      "summary": "2-3 paragraph technical summary of approach",
      "key_techniques": ["EKF sensor fusion", "RTS smoothing", "GNSS pseudorange correction"],
      "critical_decisions": [
        "Used EKF+RTS instead of particle filter",
        "Filtered satellites by C/N0 > 25",
        "Snapped final trajectory to road network"
      ]
    }
  ],
  "has_cdg": true,
  "cdg_file": "google_decimeter_1st.json"
}
```

### 1.2 Data sources for solution write-ups

**Primary sources** (structured, high-quality):
- Kaggle competition discussion forums (winner announcements)
- Kaggle solution notebooks (public kernels from top teams)
- GitHub repos linked from winner posts

**Secondary sources** (need curation):
- Blog posts by competition winners
- Papers published from competition solutions (e.g., NeurIPS competition track)
- YouTube/conference presentations by winners

**Existing coverage** (from our 125 CDGs):
- We already have 125 competition CDGs with solution summaries
- These cover the winning approach for each competition
- The CDG `audit.references` field links to source URLs

**Gap to fill**:
- Extract the competition PROMPT (problem description, data description,
  metric) for each of our 125 CDGs — this is what the architect receives
- For competitions not in our CDG library, scrape new prompts+solutions

### 1.3 Compilation process

**Step 1: Extract prompts for existing 125 CDGs** → Coding agent

For each CDG in `sciona-atoms/data/solution_cdgs/`, the competition prompt
can be reconstructed from:
- `summary` + `dejargonized_summary` → what the problem is
- `inputs` → what data is available
- `applicability.use_when` → problem characteristics
- `audit.references` → source URLs for the original competition page

Create a script that generates a `competition_prompts.json` with the
architect-facing prompt for each competition. The prompt should be
what a user would naturally provide: "I have [data]. I need to predict
[target]. The metric is [metric]. The data has [characteristics]."

**Step 2: Scrape additional competitions** → Deep research

Use deep research to compile prompts and solution summaries for major
Kaggle competitions NOT in our 125-CDG library. Priority targets:

- Featured competitions with prize money > $10K (higher solution quality)
- Competitions with public 1st-place write-ups
- Diverse problem types not well-represented in current library

Candidate sources:
- `kaggle.com/competitions?sortBy=prize` for high-value competitions
- `kaggle.com/search?q=1st+place+solution` for available write-ups
- The Kaggle Meta dataset (competition metadata)

**Estimated corpus size**: 125 existing + ~50-100 new = 175-225 competitions

### 1.4 Prompt format standardization

Each competition prompt should follow this template for consistency:

```
Problem: [1-2 sentence description]
Data: [What's provided — features, modalities, sizes]
Target: [What to predict — classification, regression, ranking, etc.]
Metric: [Exact evaluation metric]
Constraints: [Time limits, compute limits, submission format]
Domain hints: [Optional — "medical imaging", "NLP", "tabular", etc.]
```

This mirrors what a real user would tell the architect.

---

## Phase 2: Run Sciona on Competition Prompts

### 2.1 Architect invocation

For each competition prompt, invoke the sciona architect pipeline:

```python
from sciona.architect import Architect
from sciona.architect.catalog import PrimitiveCatalog, seed_builtin_primitives

# Load full catalog
catalog = load_full_catalog()  # 1,837+ atoms

# Initialize architect
architect = Architect(catalog=catalog, config=default_config)

# Generate CDG from problem description
result = architect.propose(
    problem_description=prompt["description"],
    data_description=prompt["data_description"],
    metric=prompt["evaluation_metric"],
    constraints=prompt["constraints"],
)

# result.cdg → proposed solution CDG
# result.bindings → atom bindings for each stage
# result.confidence → confidence scores
# result.alternatives → alternative CDG proposals
```

### 2.2 Output capture

For each competition, save:

```json
{
  "competition_id": "...",
  "proposed_cdg": { ... },
  "proposed_bindings": { ... },
  "grounding_status": {
    "total_stages": 6,
    "fully_bound": 4,
    "approximate": 1,
    "unbound": 1,
    "orchestration": 0
  },
  "selected_template": "google_decimeter_1st or null",
  "template_match_score": 0.87,
  "wall_time_seconds": 12.3,
  "architect_reasoning": "Selected GNSS sensor fusion template because..."
}
```

### 2.3 Batch execution

Run all competitions in parallel where possible:
- Group by problem type to amortize template matching costs
- Capture both the "with CDG template library" path (Phase 1 retrieval
  finds a matching template) and the "from scratch" path (architect
  composes a novel CDG)
- Record which path was taken for each competition

### 2.4 Grounding verification

After the architect proposes a CDG, verify each stage binding:
- Run `find_matching_primitives()` for each unbound stage
- Check that bound atoms actually exist and have valid implementations
- Verify dimensional consistency if symbolic atoms are involved
- Record the grounding rate: `bound_stages / total_stages`

---

## Phase 3: Compare Against Winning Solutions

### 3.1 Comparison framework

For each competition, an evaluation agent compares the proposed CDG against
the documented winning solution on these axes:

**Axis 1: Algorithmic coverage** (most important)
- Does the proposed CDG include the critical techniques from the winning solution?
- Example: Winner used "EKF + RTS smoothing" → does the CDG have a sequential_filter stage?
- Score: fraction of `key_techniques` covered by CDG stages

**Axis 2: Pipeline topology**
- Is the overall pipeline structure similar? (preprocessing → model → postprocessing)
- Does the DAG have the right dependency structure?
- Are critical ordering constraints satisfied?

**Axis 3: Model selection**
- Did the architect choose a comparable model family?
- Example: Winner used LightGBM → did the CDG propose a gradient boosting stage?
- This is approximate — exact model choice matters less than family

**Axis 4: Novel techniques**
- Did the winning solution use techniques NOT in the CDG?
- These are the "unknown unknowns" — techniques the architect couldn't propose
  because no template or atom covers them
- Example: Data leak exploitation, competition-specific tricks

**Axis 5: Grounding completeness**
- What fraction of the proposed CDG is fully grounded (bound to atoms)?
- Unground stages are where the framework needs human intervention

### 3.2 Evaluation agent prompt

For each competition, dispatch an evaluation agent with:

```
You are comparing a sciona-proposed solution CDG against a documented
Kaggle winning solution.

Competition: [name]
Problem: [description]
Metric: [metric]

Proposed CDG:
[JSON of proposed stages, edges, bindings]

Winning solution summary:
[2-3 paragraphs from the winner's write-up]

Key techniques from winner:
[bullet list]

Evaluate on these axes:
1. Algorithmic coverage: Which key techniques are present/missing?
2. Pipeline topology: Is the overall structure reasonable?
3. Model selection: Did the architect pick a comparable model family?
4. Novel techniques: What did the winner use that the CDG missed?
5. Grounding: What fraction of stages are bound to atoms?

Output:
{
  "competition_id": "...",
  "algorithmic_coverage": 0.0-1.0,
  "covered_techniques": ["EKF", "RTS"],
  "missing_techniques": ["phone-specific bias correction"],
  "topology_reasonable": true/false,
  "model_family_match": true/false,
  "novel_techniques_missed": ["..."],
  "grounding_rate": 0.0-1.0,
  "overall_assessment": "competitive|partial|inadequate",
  "reasoning": "..."
}
```

### 3.3 Scoring rubric

| Overall assessment | Criteria |
|-------------------|----------|
| **competitive** | Covers ≥80% of key techniques, reasonable topology, model family match, grounding ≥70% |
| **partial** | Covers 50-80% of key techniques, or topology is reasonable but missing 1-2 critical stages |
| **divergent** | Covers <50% of key techniques BUT proposes a fully grounded alternative approach — trigger creative analysis |
| **inadequate** | Covers <50% of key techniques AND poorly grounded, no coherent alternative |

### 3.4 Creative divergence analysis

When sciona proposes a solution that is completely different from the
winning approach, this is NOT necessarily a failure. The cross-disciplinary
atom library may surface novel connections.

For every `divergent` result, dispatch a creative analysis agent:

```
The sciona framework proposed a solution CDG that differs substantially
from the documented winning approach.

Competition: [name]
Problem: [description]

Winning approach: [summary]
Sciona proposal: [CDG stages and bindings]

Analyze the sciona proposal:
1. Is it a coherent, plausible approach to this problem?
2. Does it leverage cross-disciplinary atoms in an interesting way?
   (e.g., applying signal processing atoms to NLP, physics atoms to finance)
3. What are its likely strengths vs the winning approach?
4. What are its likely weaknesses?
5. Could this approach be competitive if implemented well?
6. Is there a genuine cross-domain insight here?

Rate: "creative_viable" | "creative_flawed" | "incoherent"
```

These divergent results are among the most valuable outputs of the
validation — they demonstrate sciona's ability to propose non-obvious
solutions by composing atoms from different domains.

### 3.4 Aggregation

After all evaluations, compute:

```
Overall competitive rate = #competitive / #total
Technique coverage rate = mean(algorithmic_coverage)
Grounding rate = mean(grounding_rate)
Most common gaps = frequency(missing_techniques)
```

**Target**: ≥70% competitive rate across 175+ competitions

---

## Phase 4: Gap Analysis & Iteration

### 4.1 Failure analysis

For competitions rated "inadequate" or "partial":
- Classify root cause: missing template? missing atom? wrong model family?
  missing post-processing? domain-specific trick?
- Group by problem type to find systematic weaknesses

### 4.2 Template gap identification

Competitions where the architect couldn't find a matching template reveal
CDG library gaps:
- New problem types not covered by 125 templates
- Existing templates that are too specific (don't generalize)
- Templates that are too generic (miss critical domain techniques)

### 4.3 Atom gap identification

Stages in proposed CDGs that couldn't be grounded reveal atom library gaps:
- New algorithms not in the 1,837-atom library
- Existing atoms with poor retrieval discoverability
- Domain-specific operations that need research

### 4.4 Iteration plan

Based on the gap analysis:
1. Create CDG templates for uncovered problem types
2. Create atoms for ungroundable stages
3. Improve retrieval for discoverable but unfound atoms
4. Re-run the validation on the affected competitions

---

## Execution Plan

### Step 1: Build corpus (1-2 days)

| Task | Executor | Output |
|------|----------|--------|
| Extract prompts from 125 existing CDGs | Coding agent | `competition_prompts.json` |
| Deep research: 50-100 additional competitions | Deep research (batched) | Additional prompt+solution pairs |
| Standardize all prompts | Coding agent | Unified `validation_corpus.json` |

### Step 2: Run architect (1 day)

| Task | Executor | Output |
|------|----------|--------|
| Batch-invoke architect on all prompts | Coding agents (parallel) | `proposed_cdgs/` directory |
| Verify grounding for each proposal | Coding agent | Grounding reports |

### Step 3: Evaluate (1-2 days)

| Task | Executor | Output |
|------|----------|--------|
| Dispatch 5-10 parallel evaluation agents | Background agents | `evaluations/` directory |
| Aggregate scores | Coding agent | `validation_report.md` |

### Step 4: Iterate (ongoing)

| Task | Executor | Output |
|------|----------|--------|
| Failure analysis | Agent + human review | Gap classification |
| Template/atom creation | Coding agents | New CDGs + atoms |
| Re-validation | Automated | Updated scores |

### Prerequisites

- All research prompts (18-22) completed and atoms ingested
- Architect pipeline functional end-to-end (CDG proposal + binding)
- Retrieval test suite green (150+ pass)
- **High-level Python API** (see below)

### Infrastructure needed: High-Level Architect API

Currently the only entry point is the CLI. A programmatic API is needed
for batch validation. Create `sciona.api` module with:

```python
from sciona.api import Sciona

# Initialize with full atom catalog
s = Sciona.from_repos([
    "~/personal/sciona-atoms",
    "~/personal/sciona-atoms-ml",
    "~/personal/sciona-atoms-dl",
    # ... all repos
])

# Propose a solution from a problem description
result = s.propose(
    problem="Predict smartphone location from raw GNSS measurements",
    data="Pseudorange observations at 1Hz, IMU at 100Hz, ~500 traces",
    metric="50th percentile horizontal distance error (meters)",
    constraints="No external GNSS correction services at inference",
    domain_hints=["geospatial", "sensor_fusion"],  # optional
)

# Inspect the result
result.cdg          # ProposedCDG with stages, edges, bindings
result.template     # Which CDG template was selected (if any)
result.grounding    # GroundingReport: bound/unbound/orchestration counts
result.confidence   # Overall confidence score
result.alternatives # List of alternative CDG proposals
result.reasoning    # Architect's reasoning trace

# Synthesize code from a grounded CDG
code = s.synthesize(result.cdg)
code.source         # Generated Python source
code.imports        # Required imports
code.dim_check      # Dimensional consistency report
```

This API wraps the existing CLI internals:
- `PrimitiveCatalog` loading from atom repos
- CDG template retrieval via `find_matching_primitives()`
- LLM-based CDG proposal (architect agent)
- Atom binding via retrieval + LLM ranking
- Code synthesis via `SkeletonCompiler`

The API should be stateless per-call (no mutable global state) and
support batch invocation for the validation pipeline.

### Other infrastructure

- Validation corpus: `research/23_kaggle_competition_corpus.md` (research prompt)
- Script to compare proposed CDG stages against solution key_techniques
- Evaluation agent prompt template
- Creative divergence analysis agent prompt
- Aggregation and reporting scripts
