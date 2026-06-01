from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_solve_banded_system,
)

@register_atom(witness_solve_banded_system, name="solve_banded_system")
@icontract.require(lambda l_and_u, ab, b: ab.shape[1] == b.shape[0], "Precondition failed: ab.shape[1] == b.shape[0]")
@icontract.ensure(lambda result, l_and_u, ab, b: x.shape == b.shape, "Postcondition failed: x.shape == b.shape")
def solve_banded_system(l_and_u: int, ab: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Directly solve a banded linear system using LAPACK banded LU factorization.

    Args:
        l_and_u: tuple[int, int]
        ab: Banded diagonal representation matrix
        b: NDArray[np.float64]

    Returns:
        x: NDArray[np.float64]
    """
    import scipy.linalg
    return scipy.linalg.solve_banded(l_and_u=l_and_u, ab=ab, b=b) # type: ignore

