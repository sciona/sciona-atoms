from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_ransac_draw_sample,
    witness_evaluate_consensus,
)

@register_atom(witness_ransac_draw_sample, name="ransac_draw_sample")
@icontract.require(lambda X, min_samples: X.ndim == 2, "Precondition failed: X.ndim == 2")
@icontract.require(lambda X, min_samples: min_samples <= X.shape[0], "Precondition failed: min_samples <= X.shape[0]")
@icontract.ensure(lambda result, X, min_samples: len(sample_indices) == min_samples, "Postcondition failed: len(sample_indices) == min_samples")
def ransac_draw_sample(X: NDArray[np.float64], min_samples: int) -> NDArray[np.int64]:
    """Draw a random minimal sample of points, ensuring non-degeneracy.

    Args:
        X: NDArray[np.float64]
        min_samples: int

    Returns:
        sample_indices: NDArray[np.int64]
    """
    import sklearn.linear_model
    return sklearn.linear_model.RANSACRegressor(X=X, min_samples=min_samples) # type: ignore

@register_atom(witness_evaluate_consensus, name="evaluate_consensus")
@icontract.require(lambda residuals, threshold: threshold > 0.0, "Precondition failed: threshold > 0.0")
@icontract.ensure(lambda result, residuals, threshold: inlier_mask.shape == residuals.shape, "Postcondition failed: inlier_mask.shape == residuals.shape")
def evaluate_consensus(residuals: NDArray[np.float64], threshold: float) -> NDArray[np.bool_]:
    """Compare point residuals against a threshold to construct an inlier mask.

    Args:
        residuals: NDArray[np.float64]
        threshold: float

    Returns:
        inlier_mask: NDArray[np.bool_]
    """
    import sklearn.linear_model
    return sklearn.linear_model.RANSACRegressor(residuals=residuals, threshold=threshold) # type: ignore

