"""Graph node feature generation atoms.

Implements node feature construction combining one-hot category encoding
with sinusoidal positional encoding for graph neural network inputs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import witness_node_onehot_positional


@register_atom(witness_node_onehot_positional)
@icontract.require(lambda categories: categories.ndim == 1, "Categories must be 1D")
@icontract.require(lambda num_classes: num_classes >= 1, "Need at least 1 class")
@icontract.require(lambda max_position: max_position >= 1, "Max position must be >= 1")
@icontract.ensure(lambda result, categories, num_classes, encoding_dim:
                   result.shape == (len(categories), num_classes + encoding_dim),
                   "Output has one-hot + positional columns")
def node_onehot_positional(
    categories: NDArray[np.int64],
    num_classes: int,
    encoding_dim: int = 16,
    max_position: int = 1000,
) -> NDArray[np.float64]:
    """Generate node features from one-hot category encoding and sinusoidal positional encoding.

    Produces per-node feature vectors by concatenating:
    1. One-hot encoding of the category (e.g., nucleotide type)
    2. Sinusoidal positional encoding (Vaswani et al. 2017)
    """
    n = len(categories)
    # One-hot
    onehot = np.zeros((n, num_classes), dtype=np.float64)
    onehot[np.arange(n), categories] = 1.0
    # Positional encoding
    positions = np.arange(n, dtype=np.float64)
    div_term = np.exp(
        np.arange(0, encoding_dim, 2, dtype=np.float64)
        * -(np.log(max_position * 2.0) / encoding_dim)
    )
    pe = np.zeros((n, encoding_dim), dtype=np.float64)
    pe[:, 0::2] = np.sin(positions[:, None] * div_term[None, :])
    pe[:, 1::2] = np.cos(positions[:, None] * div_term[None, :pe[:, 1::2].shape[1]])
    return np.concatenate([onehot, pe], axis=1)
