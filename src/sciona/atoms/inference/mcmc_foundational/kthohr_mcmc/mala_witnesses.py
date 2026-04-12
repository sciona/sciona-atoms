from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray


def witness_mala_proposal_adjustment(*args, **kwargs) -> AbstractArray:
    return AbstractArray(shape=(1,), dtype="float64")
