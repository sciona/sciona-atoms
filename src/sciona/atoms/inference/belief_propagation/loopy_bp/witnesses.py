from __future__ import annotations
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal

_MEMO_CACHE: dict = {}


def _clear_memo_cache() -> None:
    """Clear the stored results cache so the next iteration starts fresh."""
    _MEMO_CACHE.clear()


def witness_initialize_message_passing_state(event_shape: tuple[int, ...], family: str = "normal") -> AbstractDistribution:
    """Shape-and-type check for prior init: initialize message passing state. Returns output metadata without running the real computation."""
    return AbstractDistribution(
        family=family,
        event_shape=event_shape,
    )

def witness_run_loopy_message_passing_and_belief_query(state_in: AbstractArray, v_name: AbstractArray, num_iter: AbstractArray) -> AbstractArray:
    """Shape-and-type check for message-passing: run loopy message passing and belief query. Returns output metadata without running the real computation."""
    _cache_key = ("run_loopy_message_passing_and_belief_query",)
    if _cache_key in _MEMO_CACHE:
        return _MEMO_CACHE[_cache_key]
    result = AbstractArray(shape=state_in.shape, dtype=state_in.dtype)
    _MEMO_CACHE[_cache_key] = result
    return result
