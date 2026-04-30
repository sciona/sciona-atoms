"""Pure Bayesian bandit and Bayesian-optimization acquisition atoms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta as beta_distribution
from scipy.stats import norm

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bayesian_ucb,
    witness_beta_bernoulli_update,
    witness_epsilon_greedy_select,
    witness_expected_improvement,
    witness_initialize_beta_beliefs,
    witness_probability_of_improvement,
    witness_select_best_arm,
    witness_thompson_sample_beta,
    witness_ucb_scores,
    witness_update_arm_statistics,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _as_float_array(values: FloatArray) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def _as_int_array(values: IntArray) -> IntArray:
    return np.asarray(values, dtype=np.int64)


@register_atom(witness_initialize_beta_beliefs)
@icontract.require(lambda n_arms: n_arms > 0, "n_arms must be positive")
@icontract.require(lambda prior_alpha: prior_alpha > 0.0, "prior_alpha must be positive")
@icontract.require(lambda prior_beta: prior_beta > 0.0, "prior_beta must be positive")
@icontract.ensure(lambda n_arms, result: result[0].shape == (n_arms,), "alpha vector length must match n_arms")
@icontract.ensure(lambda n_arms, result: result[1].shape == (n_arms,), "beta vector length must match n_arms")
def initialize_beta_beliefs(
    n_arms: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Create identical positive Beta prior parameters for each bandit arm."""
    return (
        np.full(n_arms, prior_alpha, dtype=np.float64),
        np.full(n_arms, prior_beta, dtype=np.float64),
    )


@register_atom(witness_beta_bernoulli_update)
@icontract.require(lambda alpha: alpha > 0.0, "alpha must be positive")
@icontract.require(lambda beta: beta > 0.0, "beta must be positive")
@icontract.require(lambda reward: reward in (0, 1), "reward must be 0 or 1")
@icontract.ensure(lambda alpha, reward, result: result[0] == alpha + reward, "alpha must add reward")
@icontract.ensure(lambda beta, reward, result: result[1] == beta + (1 - reward), "beta must add failure count")
def beta_bernoulli_update(alpha: float, beta: float, reward: int) -> tuple[float, float]:
    """Update scalar Beta posterior parameters after one Bernoulli reward."""
    return (alpha + float(reward), beta + float(1 - reward))


@register_atom(witness_thompson_sample_beta)
@icontract.require(lambda alphas, betas: alphas.shape == betas.shape, "alphas and betas must have the same shape")
@icontract.require(lambda alphas: bool(np.all(alphas > 0.0)), "alphas must be positive")
@icontract.require(lambda betas: bool(np.all(betas > 0.0)), "betas must be positive")
@icontract.require(lambda rng: isinstance(rng, np.random.Generator), "rng must be a numpy Generator")
@icontract.ensure(lambda alphas, result: result.shape == alphas.shape, "samples must match input shape")
@icontract.ensure(lambda result: bool(np.all((result >= 0.0) & (result <= 1.0))), "samples must lie in [0, 1]")
def thompson_sample_beta(alphas: FloatArray, betas: FloatArray, rng: np.random.Generator) -> FloatArray:
    """Draw one Thompson-sampling value from each independent Beta posterior."""
    alpha_values = _as_float_array(alphas)
    beta_values = _as_float_array(betas)
    return np.asarray(rng.beta(alpha_values, beta_values), dtype=np.float64)


@register_atom(witness_ucb_scores)
@icontract.require(lambda means, counts: means.shape == counts.shape, "means and counts must have the same shape")
@icontract.require(lambda means: bool(np.all(np.isfinite(means))), "means must be finite")
@icontract.require(lambda counts: bool(np.all(counts >= 0)), "counts must be non-negative")
@icontract.require(lambda total_count: total_count >= 1, "total_count must be at least one")
@icontract.require(lambda counts, total_count: total_count >= int(np.sum(counts)), "total_count must cover observed pulls")
@icontract.require(lambda c: c >= 0.0, "exploration constant must be non-negative")
@icontract.ensure(lambda means, result: result.shape == means.shape, "scores must match means shape")
@icontract.ensure(lambda counts, result: bool(np.all(np.isinf(result[counts == 0]))), "unpulled arms must score infinity")
def ucb_scores(
    means: FloatArray,
    counts: IntArray,
    total_count: int,
    c: float = 2.0**0.5,
) -> FloatArray:
    """Compute UCB1 scores with infinite scores for untried arms."""
    mean_values = _as_float_array(means)
    count_values = _as_int_array(counts)
    scores = np.full(mean_values.shape, np.inf, dtype=np.float64)
    pulled = count_values > 0
    if np.any(pulled):
        exploration = c * np.sqrt(np.log(float(total_count)) / count_values[pulled])
        scores[pulled] = mean_values[pulled] + exploration
    return scores


@register_atom(witness_bayesian_ucb)
@icontract.require(lambda alphas, betas: alphas.shape == betas.shape, "alphas and betas must have the same shape")
@icontract.require(lambda alphas: bool(np.all(alphas > 0.0)), "alphas must be positive")
@icontract.require(lambda betas: bool(np.all(betas > 0.0)), "betas must be positive")
@icontract.require(lambda quantile: 0.0 <= quantile <= 1.0, "quantile must lie in [0, 1]")
@icontract.ensure(lambda alphas, result: result.shape == alphas.shape, "scores must match input shape")
@icontract.ensure(lambda result: bool(np.all((result >= 0.0) & (result <= 1.0))), "scores must lie in [0, 1]")
def bayesian_ucb(alphas: FloatArray, betas: FloatArray, quantile: float) -> FloatArray:
    """Compute Bayes-UCB scores as Beta posterior quantiles."""
    alpha_values = _as_float_array(alphas)
    beta_values = _as_float_array(betas)
    return np.asarray(beta_distribution.ppf(quantile, alpha_values, beta_values), dtype=np.float64)


@register_atom(witness_probability_of_improvement)
@icontract.require(lambda mean, std: mean.shape == std.shape, "mean and std must have the same shape")
@icontract.require(lambda mean: bool(np.all(np.isfinite(mean))), "mean values must be finite")
@icontract.require(lambda std: bool(np.all(std >= 0.0)), "std values must be non-negative")
@icontract.require(lambda xi: xi >= 0.0, "xi must be non-negative")
@icontract.ensure(lambda mean, result: result.shape == mean.shape, "probabilities must match input shape")
@icontract.ensure(lambda result: bool(np.all((result >= 0.0) & (result <= 1.0))), "probabilities must lie in [0, 1]")
def probability_of_improvement(
    mean: FloatArray,
    std: FloatArray,
    best_observed: float,
    xi: float = 0.0,
) -> FloatArray:
    """Compute the chance that Gaussian predictions exceed the incumbent value."""
    mean_values = _as_float_array(mean)
    std_values = _as_float_array(std)
    improvement = mean_values - best_observed - xi
    out = np.zeros(mean_values.shape, dtype=np.float64)
    positive_std = std_values > 0.0
    out[positive_std] = norm.cdf(improvement[positive_std] / std_values[positive_std])
    out[~positive_std] = (improvement[~positive_std] > 0.0).astype(np.float64)
    return out


@register_atom(witness_expected_improvement)
@icontract.require(lambda mean, std: mean.shape == std.shape, "mean and std must have the same shape")
@icontract.require(lambda mean: bool(np.all(np.isfinite(mean))), "mean values must be finite")
@icontract.require(lambda std: bool(np.all(std >= 0.0)), "std values must be non-negative")
@icontract.require(lambda xi: xi >= 0.0, "xi must be non-negative")
@icontract.ensure(lambda mean, result: result.shape == mean.shape, "expected improvements must match input shape")
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))), "expected improvements must be finite")
@icontract.ensure(lambda result: bool(np.all(result >= 0.0)), "expected improvements must be non-negative")
def expected_improvement(
    mean: FloatArray,
    std: FloatArray,
    best_observed: float,
    xi: float = 0.0,
) -> FloatArray:
    """Compute Gaussian expected improvement over the incumbent value."""
    mean_values = _as_float_array(mean)
    std_values = _as_float_array(std)
    improvement = mean_values - best_observed - xi
    out = np.maximum(improvement, 0.0)
    positive_std = std_values > 0.0
    if np.any(positive_std):
        z = improvement[positive_std] / std_values[positive_std]
        out[positive_std] = improvement[positive_std] * norm.cdf(z) + std_values[positive_std] * norm.pdf(z)
    return np.asarray(out, dtype=np.float64)


@register_atom(witness_select_best_arm)
@icontract.require(lambda scores: len(scores) > 0, "scores must be non-empty")
@icontract.require(lambda scores: bool(np.any(~np.isnan(scores))), "scores must include at least one ordered value")
@icontract.ensure(lambda scores, result: 0 <= result < len(scores), "selected arm must be a valid index")
def select_best_arm(scores: FloatArray) -> int:
    """Return the first arm index with the largest acquisition score."""
    return int(np.nanargmax(_as_float_array(scores)))


@register_atom(witness_epsilon_greedy_select)
@icontract.require(lambda values: len(values) > 0, "values must be non-empty")
@icontract.require(lambda values: bool(np.all(np.isfinite(values))), "values must be finite")
@icontract.require(lambda epsilon: 0.0 <= epsilon <= 1.0, "epsilon must lie in [0, 1]")
@icontract.require(lambda rng: isinstance(rng, np.random.Generator), "rng must be a numpy Generator")
@icontract.ensure(lambda values, result: 0 <= result < len(values), "selected arm must be a valid index")
def epsilon_greedy_select(values: FloatArray, epsilon: float, rng: np.random.Generator) -> int:
    """Select the best arm except when an explicit RNG draw triggers exploration."""
    value_array = _as_float_array(values)
    if float(rng.random()) < epsilon:
        return int(rng.integers(0, len(value_array)))
    return int(np.argmax(value_array))


@register_atom(witness_update_arm_statistics)
@icontract.require(lambda means, counts: means.shape == counts.shape, "means and counts must have the same shape")
@icontract.require(lambda means: bool(np.all(np.isfinite(means))), "means must be finite")
@icontract.require(lambda counts: bool(np.all(counts >= 0)), "counts must be non-negative")
@icontract.require(lambda means, arm: 0 <= arm < len(means), "arm must index the arrays")
@icontract.require(lambda reward: np.isfinite(reward), "reward must be finite")
@icontract.ensure(lambda means, result: result[0].shape == means.shape, "updated means must keep shape")
@icontract.ensure(lambda counts, result: result[1].shape == counts.shape, "updated counts must keep shape")
@icontract.ensure(lambda counts, arm, result: result[1][arm] == counts[arm] + 1, "selected count must increment")
def update_arm_statistics(
    means: FloatArray,
    counts: IntArray,
    arm: int,
    reward: float,
) -> tuple[FloatArray, IntArray]:
    """Update one arm's count and running mean without storing reward history."""
    updated_means = _as_float_array(means).copy()
    updated_counts = _as_int_array(counts).copy()
    updated_counts[arm] += 1
    updated_means[arm] = updated_means[arm] + (reward - updated_means[arm]) / float(updated_counts[arm])
    return updated_means, updated_counts
