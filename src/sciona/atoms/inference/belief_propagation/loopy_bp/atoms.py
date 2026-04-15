from __future__ import annotations
"""Auto-generated atom wrappers following the sciona pattern."""

from typing import TYPE_CHECKING


import numpy as np
import icontract
from sciona.ghost.registry import register_atom
from .state_models import BPState

if TYPE_CHECKING:
    import networkx as nx


# Domain-specific type aliases
from .witnesses import (
    witness_initialize_message_passing_state,
    witness_run_loopy_message_passing_and_belief_query,
)


@register_atom(witness_initialize_message_passing_state)
@icontract.require(lambda pgm: pgm is not None, "pgm cannot be None")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def initialize_message_passing_state(pgm: "nx.DiGraph", state: BPState) -> tuple[BPState, BPState]:
    """Build the immutable loopy Belief Propagation (BP) state from the Probabilistic Graphical Model (PGM).

    Args:
        pgm: Factor graph structure with variable and factor neighborhoods.
        state: Belief Propagation state containing cross-window persistent state.

    Returns:
        Initialized message-passing state and the updated BPState.
    """
    del state
    # Initialize uniform messages on all directed edges
    msg = {}
    msg_new = {}
    for node in pgm.nodes():
        msg[node] = {}
        msg_new[node] = {}
        for neighbor in pgm.neighbors(node):
            # Initialize messages as uniform distributions
            msg[node][neighbor] = np.ones(pgm.nodes[node].get('card', 2), dtype=np.float64)
            msg[node][neighbor] /= msg[node][neighbor].sum()
            msg_new[node][neighbor] = msg[node][neighbor].copy()
    new_state = BPState(pgm=pgm, msg=msg, msg_new=msg_new, t=0)
    return (new_state, new_state)


@register_atom(witness_run_loopy_message_passing_and_belief_query)
@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.require(lambda num_iter: isinstance(num_iter, int), "num_iter must be int")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def run_loopy_message_passing_and_belief_query(
    state_in: BPState, v_name: str, num_iter: int, state: BPState
) -> tuple[tuple[np.ndarray, BPState], BPState]:
    """Run loopy Belief Propagation (BP) message-passing iterations and return the queried variable's belief.

    Args:
        state_in: Immutable input snapshot with msg/msg_new/t.
        v_name: Valid variable name in the Probabilistic Graphical Model (PGM).
        num_iter: Number of message-passing iterations, >= 0.
        state: Belief Propagation state containing cross-window persistent state.

    Returns:
        Tuple of (belief, state_out) and updated BPState.
    """
    import copy
    pgm = state_in.pgm
    msg = copy.deepcopy(state_in.msg) if state_in.msg else {}
    msg_new = copy.deepcopy(state_in.msg_new) if state_in.msg_new else {}
    t = state_in.t or 0

    # Run message passing iterations
    for _ in range(num_iter):
        for node in pgm.nodes():
            for neighbor in pgm.neighbors(node):
                # Product of all incoming messages except from target
                incoming = np.ones_like(msg[node].get(neighbor, np.array([1.0])))
                for other in pgm.predecessors(node):
                    if other != neighbor and node in msg.get(other, {}):
                        incoming = incoming * msg[other][node]
                # Incorporate node potential if available
                potential = pgm.nodes[node].get('potential', None)
                if potential is not None:
                    incoming = incoming * np.asarray(potential)
                # Normalize
                s = incoming.sum()
                if s > 0:
                    incoming = incoming / s
                if node not in msg_new:
                    msg_new[node] = {}
                msg_new[node][neighbor] = incoming
        # Swap buffers
        msg = copy.deepcopy(msg_new)
        t += 1

    # Compute belief for queried variable
    belief = np.ones(pgm.nodes[v_name].get('card', 2), dtype=np.float64)
    potential = pgm.nodes[v_name].get('potential', None)
    if potential is not None:
        belief = belief * np.asarray(potential)
    for pred in pgm.predecessors(v_name):
        if pred in msg and v_name in msg[pred]:
            belief = belief * msg[pred][v_name]
    s = belief.sum()
    if s > 0:
        belief = belief / s

    new_state = BPState(pgm=pgm, msg=msg, msg_new=msg_new, t=t)
    return ((belief, new_state), new_state)
