# Research: Time Series Feature Engineering Atoms

## Goal

Find best-in-class, pure-function implementations for time series feature
extraction, aggregation, and preprocessing. Target repo: `sciona-atoms-signal`.

## CDG stages this research covers (~10 stages)

- Rolling window statistics — mean, std, min, max (Child Mind Sleep, Amex Default)
- Lag features (JPX Stock, Amex Default)
- Technical indicators — moving averages, RSI, MACD (JPX Stock, Two Sigma)
- Seasonal decomposition (existing atom — verify coverage)
- Temporal difference features — delta between timesteps (Amex Default)
- Parquet streaming ingestion concept (Child Mind Sleep, Numenta)
- Time-based entity aggregation (Amex Default, Home Credit)
- Missing value forward-fill for time series (NFL Helmet Assignment)
- Log1p transformation for skewed distributions (Web Traffic)
- Exogenous feature concatenation along time axis (Web Traffic)

## What to research

### 1. Rolling window statistics (numpy)
- `rolling_mean(values: NDArray, window: int) -> NDArray`
- `rolling_std(values: NDArray, window: int) -> NDArray`
- `rolling_min(values: NDArray, window: int) -> NDArray`
- `rolling_max(values: NDArray, window: int) -> NDArray`
- Pure numpy with uniform_filter1d or strided views
- Handle NaN / edge values

### 2. Lag features
- `create_lag_features(values: NDArray, lags: list[int]) -> NDArray`
- Shift values by each lag amount, stack as columns
- Pure numpy: `np.roll` with NaN fill

### 3. Technical indicators
- `simple_moving_average(prices: NDArray, window: int) -> NDArray`
- `exponential_moving_average(prices: NDArray, span: int) -> NDArray`
- `rsi(prices: NDArray, period: int) -> NDArray` — Relative Strength Index
- `macd(prices: NDArray, fast: int, slow: int, signal: int) -> tuple[NDArray, NDArray, NDArray]`
- `bollinger_bands(prices: NDArray, window: int, num_std: float) -> tuple[NDArray, NDArray, NDArray]`
- Source: ta-lib concepts, pure numpy implementations
- Note: check overlap with existing sciona-atoms-fintech atoms

### 4. Temporal difference
- `temporal_diff(values: NDArray, periods: int) -> NDArray`
- Simple: `values[periods:] - values[:-periods]`
- Per-entity version: diff within groups

### 5. Forward-fill for time series
- `forward_fill(values: NDArray) -> NDArray`
- Replace NaN with last valid value
- Pure numpy implementation using maximum.accumulate trick

### 6. Log1p transformation
- `log1p_transform(values: NDArray) -> NDArray`
- `expm1_inverse(values: NDArray) -> NDArray`
- Handle negative values, zeros
- np.log1p / np.expm1

### 7. Entity-grouped time aggregation
- `entity_time_aggregate(values: NDArray, entity_ids: NDArray, timestamps: NDArray, agg_fns: list[str]) -> NDArray`
- Group by entity, sort by time, compute aggregates
- Similar to tabular aggregation but time-aware

## Research questions

1. For rolling statistics: numpy uniform_filter1d vs manual strided views?
   (uniform_filter1d is simpler but doesn't handle NaN natively)
2. For technical indicators: overlap with existing sciona-atoms-fintech?
   (Check `realized_vol`, `wap`, `book_imbalance` — may already cover some)
3. For forward-fill: what's the pure numpy trick?
   (pandas ffill is easy, numpy needs accumulate + masking)
4. What contracts are natural? (window > 0, lag > 0, RSI period > 1,
   output same length as input for rolling ops)
5. Should these be in sciona-atoms-signal or a new sciona-atoms-ts repo?
   (Recommend: signal for now, separate if >30 atoms)

## Output format

Concept types: `signal_transform` for feature computation, `data_assembly` for
aggregation, and `analysis` for statistics.

For each candidate atom, provide:
```
Name: rolling_window_features
Description: Compute rolling mean, standard deviation, and extrema over a
  fixed-size time-series window.
Source: URL to the best reference implementation, paper, or library source
License: MIT, BSD, Apache-2.0, or public domain; flag any incompatible license
Concept type: signal_transform, data_assembly, or analysis
Signature: (series: NDArray, window_size: int, statistics: list[str]) -> NDArray
Pure function boundary: time-series arrays and explicit parameters in, feature
  arrays or scalar statistics out; no global state, implicit time zones,
  external services, or file I/O.
Contracts:
  - require: series is 1D or 2D with time on a documented axis
  - require: window_size > 0
  - ensure: output length and alignment are documented
Witness: short numeric sequence with window_size=3; verify rolling mean and
  standard deviation against hand-computed values.
Dependencies: numpy/pandas/scipy acceptable depending on operation and license
CDG stages covered: web_traffic/rolling_features, child_mind/sleep_features, ...
```
