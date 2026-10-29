"""Explicit ports for the fixed 43-column recipe; no statistic computation."""
import numpy as np
from sciona.ghost.registry import register_atom
from sciona.atoms.numerical.statistics.empirical_features import stack_feature_columns,witness_stack_feature_columns
from .selected_features import FEATURE_ORDER

def witness_feature_statistic_configuration() -> tuple:
    return ('adjusted_mi', 'conditional_entropy', 'conditional_entropy', 'conditional_kurtosis', 'conditional_kurtosis', 'conditional_similarity', 'conditional_similarity', 'conditional_skewness', 'conditional_skewness', 'discrete_conditional_entropy', 'discrete_conditional_entropy', 'discrete_entropy', 'discrete_entropy', 'discrete_mi', 'hsic', 'igci', 'igci', 'kurtosis', 'kurtosis', 'log_sample_count', 'log_unique', 'log_unique', 'moment21', 'moment21', 'moment31', 'moment31', 'normalized_discrete_entropy', 'normalized_discrete_entropy', 'mi_over_joint_entropy', 'mi_over_min_entropy', 'normalized_entropy', 'normalized_entropy', 'error_probability', 'error_probability', 'correlation', 'polyfit_error', 'polyfit_error', 'polyfit', 'polyfit', 'skewness', 'skewness', 'uniform_divergence', 'uniform_divergence')

@register_atom(witness_feature_statistic_configuration)
def feature_statistic_configuration() -> tuple:
    """Return one explicit statistic name per source-selected feature column."""
    return witness_feature_statistic_configuration()

def witness_assemble_selected_columns(adjusted_mi, conditional_entropy_ab, conditional_entropy_ba, conditional_kurtosis_ab, conditional_kurtosis_ba, conditional_similarity_ab, conditional_similarity_ba, conditional_skewness_ab, conditional_skewness_ba, discrete_conditional_entropy_ab, discrete_conditional_entropy_ba, discrete_entropy_a, discrete_entropy_b, discrete_mi, hsic, igci_ab, igci_ba, kurtosis_a, kurtosis_b, log_sample_count, log_unique_a, log_unique_b, moment21_ab, moment21_ba, moment31_ab, moment31_ba, normalized_discrete_entropy_a, normalized_discrete_entropy_b, mi_over_joint_entropy, mi_over_min_entropy, normalized_entropy_a, normalized_entropy_b, error_probability_ab, error_probability_ba, correlation, polyfit_error_ab, polyfit_error_ba, polyfit_ab, polyfit_ba, skewness_a, skewness_b, uniform_divergence_a, uniform_divergence_b):
    return witness_stack_feature_columns([adjusted_mi, conditional_entropy_ab, conditional_entropy_ba, conditional_kurtosis_ab, conditional_kurtosis_ba, conditional_similarity_ab, conditional_similarity_ba, conditional_skewness_ab, conditional_skewness_ba, discrete_conditional_entropy_ab, discrete_conditional_entropy_ba, discrete_entropy_a, discrete_entropy_b, discrete_mi, hsic, igci_ab, igci_ba, kurtosis_a, kurtosis_b, log_sample_count, log_unique_a, log_unique_b, moment21_ab, moment21_ba, moment31_ab, moment31_ba, normalized_discrete_entropy_a, normalized_discrete_entropy_b, mi_over_joint_entropy, mi_over_min_entropy, normalized_entropy_a, normalized_entropy_b, error_probability_ab, error_probability_ba, correlation, polyfit_error_ab, polyfit_error_ba, polyfit_ab, polyfit_ba, skewness_a, skewness_b, uniform_divergence_a, uniform_divergence_b])

@register_atom(witness_assemble_selected_columns)
def assemble_selected_columns(adjusted_mi: np.ndarray, conditional_entropy_ab: np.ndarray, conditional_entropy_ba: np.ndarray, conditional_kurtosis_ab: np.ndarray, conditional_kurtosis_ba: np.ndarray, conditional_similarity_ab: np.ndarray, conditional_similarity_ba: np.ndarray, conditional_skewness_ab: np.ndarray, conditional_skewness_ba: np.ndarray, discrete_conditional_entropy_ab: np.ndarray, discrete_conditional_entropy_ba: np.ndarray, discrete_entropy_a: np.ndarray, discrete_entropy_b: np.ndarray, discrete_mi: np.ndarray, hsic: np.ndarray, igci_ab: np.ndarray, igci_ba: np.ndarray, kurtosis_a: np.ndarray, kurtosis_b: np.ndarray, log_sample_count: np.ndarray, log_unique_a: np.ndarray, log_unique_b: np.ndarray, moment21_ab: np.ndarray, moment21_ba: np.ndarray, moment31_ab: np.ndarray, moment31_ba: np.ndarray, normalized_discrete_entropy_a: np.ndarray, normalized_discrete_entropy_b: np.ndarray, mi_over_joint_entropy: np.ndarray, mi_over_min_entropy: np.ndarray, normalized_entropy_a: np.ndarray, normalized_entropy_b: np.ndarray, error_probability_ab: np.ndarray, error_probability_ba: np.ndarray, correlation: np.ndarray, polyfit_error_ab: np.ndarray, polyfit_error_ba: np.ndarray, polyfit_ab: np.ndarray, polyfit_ba: np.ndarray, skewness_a: np.ndarray, skewness_b: np.ndarray, uniform_divergence_a: np.ndarray, uniform_divergence_b: np.ndarray) -> np.ndarray:
    """Assemble all 43 explicitly supplied columns in fixed source recipe order.

    Every numerical statistic is computed upstream; this adapter only fixes
    column order and delegates finite, aligned matrix assembly to a generic atom.
    """
    return stack_feature_columns([adjusted_mi, conditional_entropy_ab, conditional_entropy_ba, conditional_kurtosis_ab, conditional_kurtosis_ba, conditional_similarity_ab, conditional_similarity_ba, conditional_skewness_ab, conditional_skewness_ba, discrete_conditional_entropy_ab, discrete_conditional_entropy_ba, discrete_entropy_a, discrete_entropy_b, discrete_mi, hsic, igci_ab, igci_ba, kurtosis_a, kurtosis_b, log_sample_count, log_unique_a, log_unique_b, moment21_ab, moment21_ba, moment31_ab, moment31_ba, normalized_discrete_entropy_a, normalized_discrete_entropy_b, mi_over_joint_entropy, mi_over_min_entropy, normalized_entropy_a, normalized_entropy_b, error_probability_ab, error_probability_ba, correlation, polyfit_error_ab, polyfit_error_ba, polyfit_ab, polyfit_ba, skewness_a, skewness_b, uniform_divergence_a, uniform_divergence_b])

def witness_route_feature_populations(training_pairs:list,prediction_pairs:list,training_types:list,prediction_types:list) -> tuple:
    return training_pairs,prediction_pairs,training_types,prediction_types

@register_atom(witness_route_feature_populations)
def route_feature_populations(training_pairs:list,prediction_pairs:list,training_types:list,prediction_types:list) -> tuple:
    """Route independently named populations to shared generic pair interfaces."""
    return training_pairs,prediction_pairs,training_types,prediction_types
