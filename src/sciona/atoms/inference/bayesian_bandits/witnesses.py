"""Ghost witnesses for Bayesian bandit and acquisition atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_initialize_beta_beliefs(
    n_arms: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> tuple[AbstractArray, AbstractArray]:
    """Return two one-dimensional posterior-parameter arrays."""
    shape = (n_arms,)
    return (
        AbstractArray(shape=shape, dtype="float64"),
        AbstractArray(shape=shape, dtype="float64"),
    )


def witness_beta_bernoulli_update(
    alpha: float,
    beta: float,
    reward: int,
) -> tuple[float, float]:
    """Return scalar Beta posterior parameters after one binary reward."""
    return (alpha, beta)


def witness_thompson_sample_beta(
    alphas: AbstractArray,
    betas: AbstractArray,
    rng: object,
) -> AbstractArray:
    """Return one Beta sample for each arm."""
    del betas, rng
    return AbstractArray(shape=alphas.shape, dtype="float64")


def witness_ucb_scores(
    means: AbstractArray,
    counts: AbstractArray,
    total_count: int,
    c: float = 2.0**0.5,
) -> AbstractArray:
    """Return one deterministic UCB score for each arm."""
    del counts, total_count, c
    return AbstractArray(shape=means.shape, dtype="float64")


def witness_bayesian_ucb(
    alphas: AbstractArray,
    betas: AbstractArray,
    quantile: float,
) -> AbstractArray:
    """Return one posterior quantile score for each arm."""
    del betas, quantile
    return AbstractArray(shape=alphas.shape, dtype="float64")


def witness_probability_of_improvement(
    mean: AbstractArray,
    std: AbstractArray,
    best_observed: float,
    xi: float = 0.0,
) -> AbstractArray:
    """Return one improvement probability for each candidate."""
    del std, best_observed, xi
    return AbstractArray(shape=mean.shape, dtype="float64")


def witness_expected_improvement(
    mean: AbstractArray,
    std: AbstractArray,
    best_observed: float,
    xi: float = 0.0,
) -> AbstractArray:
    """Return one expected-improvement value for each candidate."""
    del std, best_observed, xi
    return AbstractArray(shape=mean.shape, dtype="float64")


def witness_select_best_arm(scores: AbstractArray) -> int:
    """Return a scalar index into the score vector."""
    del scores
    return 0


def witness_epsilon_greedy_select(
    values: AbstractArray,
    epsilon: float,
    rng: object,
) -> int:
    """Return a scalar index chosen from the value vector."""
    del values, epsilon, rng
    return 0


def witness_update_arm_statistics(
    means: AbstractArray,
    counts: AbstractArray,
    arm: int,
    reward: float,
) -> tuple[AbstractArray, AbstractArray]:
    """Return updated mean and count arrays with unchanged shapes."""
    del arm, reward
    return (
        AbstractArray(shape=means.shape, dtype="float64"),
        AbstractArray(shape=counts.shape, dtype="int64"),
    )
