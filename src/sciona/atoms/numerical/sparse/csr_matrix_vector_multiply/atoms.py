from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_validate_spmv_shapes,
    witness_compute_spmv_kernel,
)

@register_atom(witness_validate_spmv_shapes, name="validate_spmv_shapes")
@icontract.require(lambda indptr, x: indptr.ndim == 1, "Precondition failed: indptr.ndim == 1")
@icontract.require(lambda indptr, x: x.ndim == 1, "Precondition failed: x.ndim == 1")
@icontract.ensure(lambda result, indptr, x: result is not None, "Postcondition failed: result is not None")
def validate_spmv_shapes(indptr: NDArray[np.int64], x: NDArray[np.float64]) -> bool:
    """Ensure matrix column count matches the input multiplier vector size.

    Args:
        indptr: NDArray[np.int64]
        x: NDArray[np.float64]

    Returns:
        is_valid: bool
    """
    import scipy.sparse.base.spmatrix
    return scipy.sparse.base.spmatrix._check_vector_shape(indptr=indptr, x=x) # type: ignore

@register_atom(witness_compute_spmv_kernel, name="compute_spmv_kernel")
@icontract.require(lambda indptr, indices, data, x: indices.shape[0] == data.shape[0], "Precondition failed: indices.shape[0] == data.shape[0]")
@icontract.ensure(lambda result, indptr, indices, data, x: y.shape[0] == indptr.shape[0] - 1, "Postcondition failed: y.shape[0] == indptr.shape[0] - 1")
def compute_spmv_kernel(indptr: NDArray[np.int64], indices: NDArray[np.int64], data: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Perform sparse matrix-vector dot product operations.

    Args:
        indptr: NDArray[np.int64]
        indices: NDArray[np.int64]
        data: NDArray[np.float64]
        x: NDArray[np.float64]

    Returns:
        y: NDArray[np.float64]
    """
    import scipy.sparse._sparsetools
    return scipy.sparse._sparsetools.csr_matvec(indptr=indptr, indices=indices, data=data, x=x) # type: ignore

