from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_assemble_collocation_system(fun: AbstractArray, bc: AbstractArray, x_mesh: AbstractArray, y_guess: AbstractArray) -> AbstractArray:
    """Ghost witness for assemble_collocation_system."""
    _ = (fun, bc, x_mesh, y_guess)
    return AbstractArray(shape=fun.shape, dtype=fun.dtype)

def witness_solve_bvp_newton_step(y_guess: AbstractArray, residuals: AbstractArray, jacobian: AbstractScalar | Any) -> AbstractArray:
    """Ghost witness for solve_bvp_newton_step."""
    _ = (y_guess, residuals, jacobian)
    return AbstractArray(shape=y_guess.shape, dtype=y_guess.dtype)

def witness_evaluate_collocation_residuals(fun: AbstractArray, x_mesh: AbstractArray, y_sol: AbstractArray) -> AbstractArray:
    """Ghost witness for evaluate_collocation_residuals."""
    _ = (fun, x_mesh, y_sol)
    return AbstractArray(shape=fun.shape, dtype=fun.dtype)

def witness_refine_bvp_mesh_nodes(x_mesh: AbstractArray, y_sol: AbstractArray, interval_residuals: AbstractArray, tol: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for refine_bvp_mesh_nodes."""
    _ = (x_mesh, y_sol, interval_residuals, tol)
    return AbstractArray(shape=x_mesh.shape, dtype=x_mesh.dtype)

