# Research: Bayesian Optimization & Multi-Armed Bandit Atoms

## Goal

Find best-in-class, pure-function implementations for Bayesian optimization,
multi-armed bandit algorithms, and Thompson sampling. Target repo: `sciona-atoms`.

## CDG stages this research covers (~8 stages)

- Beta distribution belief state initialization (Santa 2020 Candy Cane)
- Bayesian UCB sampling (Santa 2020 Candy Cane)
- Thompson sampling from Beta posteriors (Santa 2020)
- Belief updating — Beta-Bernoulli conjugate update (Santa 2020)
- Action execution / arm selection (Santa 2020)
- Expected improvement acquisition function (general Bayesian optimization)
- Probability of improvement acquisition function (general)
- UCB acquisition function (general)

## What to research

### 1. Beta-Bernoulli conjugate update
- `beta_update(alpha: float, beta: float, reward: int) -> tuple[float, float]`
- Simple: alpha += reward, beta += (1 - reward)
- Pure Python, trivial but important as a composable atom

### 2. Thompson sampling from Beta
- `thompson_sample_beta(alphas: NDArray, betas: NDArray) -> NDArray`
- Draw from Beta(alpha, beta) for each arm, return samples
- Source: numpy.random.beta

### 3. UCB (Upper Confidence Bound)
- `ucb_scores(means: NDArray, counts: NDArray, total_count: int, c: float) -> NDArray`
- Formula: `mean + c * sqrt(log(total_count) / count)`
- UCB1 variant

### 4. Bayesian UCB
- `bayesian_ucb(alphas: NDArray, betas: NDArray, quantile: float) -> NDArray`
- Use Beta distribution quantile (ppf) as UCB
- `scipy.stats.beta.ppf(quantile, alphas, betas)`

### 5. Expected Improvement (EI)
- `expected_improvement(mean: NDArray, std: NDArray, best_observed: float, xi: float) -> NDArray`
- EI = (mean - best - xi) * Phi(Z) + std * phi(Z) where Z = (mean - best - xi) / std
- Source: scipy.stats.norm (BSD)

### 6. Probability of Improvement (PI)
- `probability_of_improvement(mean: NDArray, std: NDArray, best_observed: float, xi: float) -> NDArray`
- PI = Phi((mean - best - xi) / std)

### 7. Epsilon-greedy selection
- `epsilon_greedy_select(values: NDArray, epsilon: float, rng: np.random.Generator) -> int`
- With probability epsilon, select random arm; otherwise select argmax

### 8. Arm selection and tracking
- `select_best_arm(scores: NDArray) -> int`
- `update_arm_statistics(means: NDArray, counts: NDArray, arm: int, reward: float) -> tuple[NDArray, NDArray]`
- Incremental mean update

## Research questions

1. For Thompson sampling: what's the numerically stable Beta sampling?
   (numpy.random.beta handles edge cases with alpha/beta near 0)
2. For EI: what's the standard implementation?
   (Snoek et al. 2012 — straightforward with scipy.stats.norm)
3. For UCB: UCB1 vs KL-UCB vs Bayesian UCB — which variants are most useful?
   (UCB1 for simplicity, Bayesian UCB for Beta priors)
4. What contracts are natural? (alpha > 0, beta > 0, epsilon in [0,1],
   counts > 0 for UCB, std > 0 for EI)
5. Should the random number generation be explicit?
   (Yes — pass rng/seed for reproducibility)

## Output format

Concept types: `posterior_update` for belief updates, `probabilistic_oracle` for
acquisition functions, and `sampler` for Thompson sampling.

For each candidate atom, provide:
```
Name: beta_bernoulli_update
Description: Update Beta posterior parameters after Bernoulli successes and
  failures.
Source: URL to the best reference implementation, paper, or library source
License: MIT, BSD, Apache-2.0, or public domain; flag any incompatible license
Concept type: posterior_update, probabilistic_oracle, or sampler
Signature: (alpha: float, beta: float, successes: int, failures: int) -> tuple[float, float]
Pure function boundary: posterior parameters, observations, and explicit RNG or
  seed when sampling in; updated parameters, acquisition scores, or samples out;
  no hidden global RNG, optimizer state, network calls, or file I/O.
Contracts:
  - require: alpha > 0 and beta > 0
  - require: successes >= 0 and failures >= 0
  - ensure: updated_alpha == alpha + successes
  - ensure: updated_beta == beta + failures
Witness: alpha=1, beta=1, successes=3, failures=2 returns (4, 3).
Dependencies: numpy/scipy preferred; heavier Bayesian optimization libraries
  acceptable only as reference sources, not required runtime dependencies
CDG stages covered: santa_workshop/bandit_selection, optimization/acquisition, ...
```
