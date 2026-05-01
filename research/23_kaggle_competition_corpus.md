# Research: Kaggle Competition Corpus — Prompts & Winning Solutions

## Goal

Compile a comprehensive corpus of Kaggle competitions that have publicly
available 1st-place solution write-ups. For each competition, extract the
problem description (what a user would tell sciona) and a structured summary
of the winning approach.

This corpus will be used to validate the sciona framework end-to-end:
feed the problem description into the architect, generate a CDG, ground it
with atoms, and compare against the documented winning solution.

## What to collect

For EVERY Kaggle competition where a 1st-place solution write-up exists
(discussion post, blog, GitHub, paper), collect:

### Competition metadata
- `competition_id`: Kaggle URL slug (e.g., "google-smartphone-decimeter-2021")
- `title`: Full competition title
- `year`: Competition year
- `host`: Competition host organization
- `prize`: Total prize pool (if applicable)
- `participants`: Number of teams
- `problem_type`: classification, regression, ranking, segmentation, detection,
  recommendation, optimization, generation, or other
- `domain`: primary domain (tabular, cv, nlp, audio, medical, geospatial,
  time_series, graph, recommender, reinforcement_learning, scientific, other)
- `modalities`: list of data modalities (tabular, image, text, audio, video,
  3d_volume, graph, time_series, geospatial)

### Problem prompt (what a user would provide to sciona)
- `description`: 2-4 sentence natural language problem statement
- `data_description`: What data is provided — feature types, formats, sizes,
  number of samples, special characteristics (class imbalance, missing values,
  multi-modal, temporal, etc.)
- `evaluation_metric`: Exact metric name and formula if non-standard
  (e.g., "QWK", "CRPS", "MAP@5", "Dice coefficient", "log loss")
- `constraints`: Submission format, time limits, compute limits, external
  data rules
- `key_challenges`: What makes this problem hard (e.g., "extreme class
  imbalance", "noisy labels", "domain shift between train/test",
  "very large dataset", "multi-modal fusion required")

### Winning solution summary
- `placement`: "1st" (also collect 2nd/3rd if they used notably different approaches)
- `team`: Team or individual name
- `source_url`: URL to the write-up (discussion post, blog, GitHub)
- `source_type`: "kaggle_discussion", "github_repo", "blog_post", "paper"
- `summary`: 2-3 paragraph technical summary covering:
  - Overall pipeline architecture
  - Key modeling choices (model family, architecture, loss function)
  - Critical preprocessing and feature engineering steps
  - Ensemble strategy
  - Any tricks or insights that were decisive
- `key_techniques`: List of 5-10 specific techniques used, e.g.:
  ["5-fold stratified CV", "EfficientNet-B4 backbone", "CutMix augmentation",
   "pseudo-labeling on test data", "TTA with horizontal flip",
   "rank-average ensemble of 3 models"]
- `critical_decisions`: List of 3-5 decisions that differentiated the winning
  solution, e.g.:
  ["Used EKF instead of particle filter — 10x faster with similar accuracy",
   "Applied Ben Graham preprocessing for retinal images",
   "Trained on external data (not just competition data)"]
- `novel_insights`: Any insights that would be surprising or non-obvious,
  even to an experienced practitioner

## Scope

### Must include (our existing 125 CDGs)
We already have CDGs for 125 competitions. For each, verify we have the
competition prompt in the right format. The CDG files contain solution
summaries but may not have the PROBLEM DESCRIPTION (what the user sees
before knowing the solution).

List of our 125 competitions: see `sciona-atoms/data/solution_cdgs/*.json`
(exclude `_bindings.json` files).

### New competitions to add
Search for ALL additional Kaggle competitions with public 1st-place
write-ups. Priority order:

1. **Featured competitions** (hosted by companies, with prizes)
   - These have the highest-quality solutions and write-ups
   - Source: `kaggle.com/competitions?hostSegmentIdFilter=1`

2. **Research competitions** (hosted by research orgs)
   - Often have novel problem formulations
   - Source: `kaggle.com/competitions?hostSegmentIdFilter=2`

3. **Community competitions** with significant participation (>500 teams)
   - Quality varies but volume is high
   - Filter by team count

4. **Playground competitions** with novel problem types
   - Good for testing framework generalization

### Where to find 1st-place write-ups

1. **Kaggle discussion forums**: Search each competition's discussion tab
   for posts by top-placing teams. Common patterns:
   - Title contains "1st place", "gold", "winning"
   - Posted by the competition winner
   - Often pinned by competition hosts

2. **GitHub**: Many winners publish code repos
   - Search: `kaggle 1st place solution site:github.com`
   - Check the winner's Kaggle profile for linked GitHub

3. **Blog posts**: Winners often write detailed blog posts
   - Medium, personal blogs, company tech blogs
   - Search: `"kaggle" "1st place" "solution" site:medium.com`

4. **Papers**: Some competition solutions are published as papers
   - NeurIPS competition track, CVPR workshops, KDD cup papers
   - Search: arxiv, Google Scholar

5. **Existing compilations**:
   - https://github.com/interviewBubble/Kaggle-Solutions (check if current)
   - https://farid.one/kaggle-solutions/ (comprehensive list)
   - https://www.kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions

## Output format

Produce a single JSON file `validation_corpus.json` with this structure:

```json
{
  "metadata": {
    "compiled_date": "2026-05-01",
    "total_competitions": 225,
    "existing_cdg_count": 125,
    "new_additions": 100,
    "sources": ["kaggle_discussions", "github", "blogs", "papers"]
  },
  "competitions": [
    {
      "competition_id": "google-smartphone-decimeter-2021",
      "title": "Google Smartphone Decimeter Challenge",
      "year": 2021,
      "host": "Google",
      "prize": 10000,
      "participants": 810,
      "problem_type": "regression",
      "domain": "geospatial",
      "modalities": ["tabular", "time_series", "geospatial"],
      "has_existing_cdg": true,
      "cdg_file": "google_decimeter_1st.json",
      "prompt": {
        "description": "Predict the precise location of a smartphone using raw GNSS measurements. Given sequences of satellite pseudorange observations and IMU sensor data, output latitude/longitude coordinates accurate to sub-meter precision.",
        "data_description": "Raw GNSS measurements (pseudorange, carrier phase, C/N0, constellation) at 1Hz + IMU data (accelerometer, gyroscope, magnetometer) at 100Hz. ~500 driving traces, each 5-30 minutes. Ground truth from NovAtel SPAN reference system.",
        "evaluation_metric": "50th percentile of horizontal distance error in meters",
        "constraints": "Submission is lat/lon per millisecond epoch. No external GNSS correction services allowed at inference time.",
        "key_challenges": ["Multipath interference in urban canyons", "Clock bias estimation", "Sensor fusion of GNSS + IMU", "Variable satellite visibility"]
      },
      "winning_solutions": [
        {
          "placement": "1st",
          "team": "Team name",
          "source_url": "https://www.kaggle.com/c/google-smartphone-decimeter/discussion/...",
          "source_type": "kaggle_discussion",
          "summary": "Applied Extended Kalman Filter with IMU-assisted prediction...",
          "key_techniques": [
            "Extended Kalman Filter (EKF) for GNSS+IMU fusion",
            "Rauch-Tung-Striebel (RTS) backward smoother",
            "C/N0-based satellite signal quality filtering",
            "Snap-to-road-network post-processing",
            "Phone-specific clock bias correction"
          ],
          "critical_decisions": [
            "Used EKF over particle filter for computational efficiency",
            "Filtered satellites with C/N0 < 25 dB-Hz",
            "Applied road network matching as final step"
          ],
          "novel_insights": [
            "Phone-specific systematic biases are the dominant error source, not GNSS accuracy"
          ]
        }
      ]
    }
  ]
}
```

## Research questions

1. How many Kaggle competitions have public 1st-place write-ups?
   (Estimate: 300-500 featured + research competitions since 2015)
2. What is the distribution by problem type and domain?
3. Are there competitions where top solutions are fundamentally different
   from each other? (These are interesting for testing sciona's flexibility)
4. What competitions have NO public write-up? (These are less useful for
   validation but worth noting for coverage)
5. For our existing 125 CDGs, do we have the PROBLEM DESCRIPTION as well
   as the solution? (Many CDGs were built from solution write-ups and may
   not include the original problem statement)

## Notes for the research agent

- Be thorough: aim for 200+ competitions total
- Focus on QUALITY of the prompt and solution summary over quantity
- For the problem prompt: write it as a USER would describe it to an AI
  assistant, NOT as the competition host describes it (remove Kaggle-specific
  jargon like "kernel", "submission deadline", etc.)
- For key_techniques: be SPECIFIC — "EfficientNet-B4" not just "CNN",
  "CutMix" not just "augmentation", "5-fold stratified CV" not just "CV"
- Include the source URL for every solution so we can verify
- Flag competitions where the winning approach is particularly creative
  or unconventional — these are the most interesting test cases for sciona
