from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_assemble_collocation_system,
    witness_solve_bvp_newton_step,
    witness_evaluate_collocation_residuals,
    witness_refine_bvp_mesh_nodes,
)

@register_atom(witness_assemble_collocation_system, name="assemble_collocation_system")
@icontract.require(lambda fun, bc, x_mesh, y_guess: len(x_mesh) >= 2, "Precondition failed: len(x_mesh) >= 2")
@icontract.ensure(lambda result, fun, bc, x_mesh, y_guess: result is not None, "Postcondition failed: result is not None")
def assemble_collocation_system(fun: Callable[[float, NDArray[np.float64]], NDArray[np.float64]], bc: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]], x_mesh: NDArray[np.float64], y_guess: NDArray[np.float64]) -> NDArray[np.float64]:
    """Construct the global non-linear algebraic system and sparse Jacobian for collocation and boundary conditions.

    Args:
        fun: Callable[[float, NDArray[np.float64]], NDArray[np.float64]]
        bc: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        x_mesh: NDArray[np.float64]
        y_guess: NDArray[np.float64]

    Returns:
        residuals: NDArray[np.float64]
    """
    import scipy.integrate
    return scipy.integrate.solve_bvp(fun=fun, bc=bc, x_mesh=x_mesh, y_guess=y_guess) # type: ignore

@register_atom(witness_solve_bvp_newton_step, name="solve_bvp_newton_step")
@icontract.require(lambda y_guess, residuals, jacobian: y_guess is not None, "Precondition failed: y_guess is not None")
@icontract.ensure(lambda result, y_guess, residuals, jacobian: result is not None, "Postcondition failed: result is not None")
def solve_bvp_newton_step(y_guess: NDArray[np.float64], residuals: NDArray[np.float64], jacobian: Any) -> NDArray[np.float64]:
    """Perform sparse Newton-Raphson updates to solve collocation equations.

    Args:
        y_guess: NDArray[np.float64]
        residuals: NDArray[np.float64]
        jacobian: scipy.sparse.coo_matrix

    Returns:
        y_new: NDArray[np.float64]
    """
    import scipy.sparse.linalg
    return scipy.sparse.linalg.spsolve(y_guess=y_guess, residuals=residuals, jacobian=jacobian) # type: ignore

@register_atom(witness_evaluate_collocation_residuals, name="evaluate_collocation_residuals")
@icontract.require(lambda fun, x_mesh, y_sol: fun is not None, "Precondition failed: fun is not None")
@icontract.ensure(lambda result, fun, x_mesh, y_sol: len(interval_residuals) == len(x_mesh) - 1, "Postcondition failed: len(interval_residuals) == len(x_mesh) - 1")
def evaluate_collocation_residuals(fun: Callable[[float, NDArray[np.float64]], NDArray[np.float64]], x_mesh: NDArray[np.float64], y_sol: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute localized differential equation residual on each mesh interval.

    Args:
        fun: Callable[[float, NDArray[np.float64]], NDArray[np.float64]]
        x_mesh: NDArray[np.float64]
        y_sol: NDArray[np.float64]

    Returns:
        interval_residuals: NDArray[np.float64]
    """
    import scipy.integrate
    return scipy.integrate.solve_bvp(fun=fun, x_mesh=x_mesh, y_sol=y_sol) # type: ignore

@register_atom(witness_refine_bvp_mesh_nodes, name="refine_bvp_mesh_nodes")
@icontract.require(lambda x_mesh, y_sol, interval_residuals, tol: x_mesh is not None, "Precondition failed: x_mesh is not None")
@icontract.ensure(lambda result, x_mesh, y_sol, interval_residuals, tol: result is not None, "Postcondition failed: result is not None")
def refine_bvp_mesh_nodes(x_mesh: NDArray[np.float64], y_sol: NDArray[np.float64], interval_residuals: NDArray[np.float64], tol: float) -> NDArray[np.float64]:
    """Insert new nodes in intervals where local residuals exceed the target tolerances.

    Args:
        x_mesh: NDArray[np.float64]
        y_sol: NDArray[np.float64]
        interval_residuals: NDArray[np.float64]
        tol: float

    Returns:
        new_x_mesh: NDArray[np.float64]
    """
    import scipy.integrate
    return scipy.integrate.solve_bvp(x_mesh=x_mesh, y_sol=y_sol, interval_residuals=interval_residuals, tol=tol) # type: ignore

