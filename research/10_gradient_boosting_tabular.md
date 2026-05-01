# Research: Gradient Boosting & Tabular ML Atoms

## Goal

Find best-in-class, pure-function implementations for tabular ML preprocessing
and gradient boosting utilities. These are NOT model wrappers — they are the
pure-function components around gradient boosting pipelines.
Target repo: `sciona-atoms-ml`.

## CDG stages this research covers (~11 stages)

- LightGBM/XGBoost training with CV (Porto Seguro, Instacart, many tabular CDGs)
- Feature interaction creation (IEEE CIS Fraud, Otto Group Product)
- Target encoding for high-cardinality categoricals (PetFinder, IEEE CIS Fraud)
- Time-decay weighting for temporal features (Home Credit Default)
- Frequency encoding (IEEE CIS Fraud, Otto Group)
- Entity aggregation — group-by statistics (Amex Default, Instacart)
- Missing value indicators and imputation strategies (Porto Seguro)
- Rank transformation (JPX Stock, Porto Seguro)
- Feature importance-based selection (MoA Prediction)
- Pseudo-label feature selection (MoA Prediction)
- Multi-table join and aggregation (Home Credit Default, Instacart)

## What to research

### 1. Feature interaction atoms
- Pairwise multiplication: `pairwise_products(features: NDArray) -> NDArray`
- Pairwise ratios: `pairwise_ratios(features: NDArray, epsilon: float) -> NDArray`
- Group-by statistics: `group_aggregate(values: NDArray, groups: NDArray, agg_fn: str) -> NDArray`
  (mean, std, min, max, count, last, first)
- Source: pure numpy/pandas operations

### 2. Frequency encoding
- Replace each category with its frequency in the training set
- `frequency_encode(categories: NDArray, frequency_map: dict) -> NDArray`
- `frequency_encode_fit(categories: NDArray) -> dict`
- Pure Python/numpy

### 3. Time-decay weighting
- Apply exponential decay to aggregate features by recency
- `time_decay_aggregate(values: NDArray, timestamps: NDArray, decay_rate: float) -> NDArray`
- More recent observations get higher weight

### 4. Rank transformation
- Replace values with their rank (scipy.stats.rankdata style)
- `rank_transform(values: NDArray, method: str) -> NDArray`
- Methods: average, min, max, dense, ordinal
- Source: scipy.stats.rankdata (BSD)

### 5. Multi-table aggregation
- Aggregate child table to parent table via group-by
- `aggregate_child_table(child_values: NDArray, parent_keys: NDArray, child_keys: NDArray, agg_fns: list[str]) -> NDArray`
- Common in relational tabular competitions (Home Credit, Instacart)

### 6. Difference features
- Compute delta between consecutive time steps per entity
- `temporal_difference(values: NDArray, entity_ids: NDArray, sort_keys: NDArray) -> NDArray`
- Last value minus second-to-last value
- Amex Default uses this heavily

### 7. Rolling window statistics
- Rolling mean, std, min, max over entity-specific time windows
- `rolling_statistics(values: NDArray, window_size: int, agg_fns: list[str]) -> NDArray`
- Pure numpy with stride tricks

### 8. Gradient boosting loss decomposition
- LightGBM/XGBoost custom objective interface
- Separate the objective function and gradient computation from the tree building
- `tweedie_gradient(predictions: NDArray, targets: NDArray, power: float) -> tuple[NDArray, NDArray]`
  returns (gradient, hessian)
- Note: we already have `estimate_tweedie_power` — research what other
  gradient/hessian decompositions are useful

## Research questions

1. What are the pure numpy implementations for group-by aggregations?
   (Without pandas — or define pandas as acceptable dependency?)
2. For feature interactions: what's the memory-efficient way to compute
   pairwise products for wide feature matrices?
3. For time-decay: what decay functions are used in practice?
   (Exponential, linear, log — exponential is most common)
4. What contracts are natural? (decay_rate > 0, window_size > 0,
   rank output has same shape as input, frequency values non-negative)
5. Should gradient boosting model training be an opaque atom?
   (Yes — the tree building is compiled C++. Atom should cover
   preprocessing and post-processing around it.)

## Output format

Concept types: `data_assembly` for aggregation, `signal_transform` for feature
transforms, and `analysis` for statistics.

For each candidate atom, provide:
```
Name: target_encode
Description: Compute smoothed target encoding values for categorical features.
Source: URL to the best reference implementation, paper, or library source
License: MIT, BSD, Apache-2.0, or public domain; flag any incompatible license
Concept type: data_assembly, signal_transform, or analysis
Signature: (categories: NDArray, targets: NDArray, smoothing: float) -> dict
Pure function boundary: tabular arrays and explicit parameters in, encoded
  arrays or mapping objects out; no model fitting side effects, global state,
  random state unless explicitly passed, or file I/O.
Contracts:
  - require: len(categories) == len(targets)
  - require: smoothing >= 0
  - ensure: every output encoding is finite
Witness: small categorical column with repeated categories; verify smoothed
  means against hand-computed values.
Dependencies: numpy/pandas/sklearn acceptable when justified; avoid depending on
  XGBoost/LightGBM internals for preprocessing atoms
CDG stages covered: home_credit/target_encoding, santander/feature_statistics, ...
```
