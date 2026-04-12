"""Auto-generated Pydantic state models for cross-window state."""

from __future__ import annotations

from typing import Any

import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore

from pydantic import BaseModel, ConfigDict, Field

class BPState(BaseModel):
    """Immutable loopy-belief-propagation state: graph structure/tables plus current and next directed message buffers and iteration index. Invariants: all message vectors are nonnegative and normalized (sum to 1), buffers cover all variable<->factor directed edges, and t >= 0 with transitions producing a new BPState value instead of mutating prior state."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pgm: Any | None = Field(default=None)
    msg: dict[str, dict[str, Any]] | None = Field(default=None)
    msg_new: dict[str, dict[str, Any]] | None = Field(default=None)
    t: int | None = Field(default=None)
