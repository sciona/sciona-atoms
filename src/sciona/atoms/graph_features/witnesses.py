"""Ghost witnesses for graph feature generation atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_node_onehot_positional(
    categories: AbstractArray,
    num_classes: int,
    encoding_dim: int = 16,
    max_position: int = 1000,
) -> AbstractArray:
    """Ghost witness for one-hot plus positional encoding node features.

    Takes a 1-D category array and returns a 2-D feature array of shape
    (len(categories), num_classes + encoding_dim).
    """
    return AbstractArray()
