from __future__ import annotations

from .atoms import (
    compute_gram_matrix,
    apply_tikhonov_shift_and_solve,
)

__all__ = [
    "compute_gram_matrix",
    "apply_tikhonov_shift_and_solve",
]
