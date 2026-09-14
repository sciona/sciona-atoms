"""Finite, broadcastable equation residuals; these functions do not solve equations.

A zero residual establishes arithmetic consistency only. Physical interpretation
requires an independently applicable ohmic/series model. No resistance-sign or
nonzero-current guard is imposed by a residual evaluator. Scientific reference:
https://openstax.org/books/university-physics-volume-2/pages/10-2-resistors-in-series-and-parallel
"""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.registry import register_atom

V = DimensionalSignature(M=1,L=2,T=-3,I=-1)
R = DimensionalSignature(M=1,L=2,T=-3,I=-2)
I = DimensionalSignature(I=1)


def _arrays(*arguments):
    values=[]
    for argument in arguments:
        value=np.asarray(argument)
        if value.dtype.kind not in 'iuf':raise ValueError('finite real numeric inputs required')
        value=value.astype(np.float64)
        if not np.all(np.isfinite(value)):raise ValueError('finite real numeric inputs required')
        values.append(value)
    return np.broadcast_arrays(*values)


def _result(value):
    value=np.asarray(value,dtype=np.float64)
    if not np.all(np.isfinite(value)):raise ValueError('finite output required')
    return value


def _witness(*arguments):
    return AbstractArray(shape=np.broadcast_shapes(*(arg.shape for arg in arguments)),dtype='float64')


def witness_ohmic(voltage: AbstractArray,current: AbstractArray,resistance: AbstractArray) -> AbstractArray:
    return _witness(voltage,current,resistance)


def witness_voltage(total_voltage: AbstractArray,voltage_a: AbstractArray,voltage_b: AbstractArray) -> AbstractArray:
    return _witness(total_voltage,voltage_a,voltage_b)


def witness_resistance(total_resistance: AbstractArray,resistance_a: AbstractArray,resistance_b: AbstractArray) -> AbstractArray:
    return _witness(total_resistance,resistance_a,resistance_b)


def witness_balance(current: AbstractArray,total_resistance: AbstractArray,resistance_a: AbstractArray,resistance_b: AbstractArray) -> AbstractArray:
    return _witness(current,total_resistance,resistance_a,resistance_b)


@register_atom(witness_ohmic,dim_map={'voltage':V,'current':I,'resistance':R,'return':V})
def ohmic_voltage_residual(voltage: NDArray[np.float64],current: NDArray[np.float64],resistance: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return voltage minus current times resistance, in volts.

    Inputs are finite real volts, amperes and ohms; broadcast to float64 arrays.
    Reject nonfinite values, incompatible shapes and floating-point errors.
    Inputs are not mutated. Zero is consistency, not physical validation.
    """
    v,i,r=_arrays(voltage,current,resistance)
    with np.errstate(all='raise'):return _result(v-i*r)


@register_atom(witness_voltage,dim_map={'total_voltage':V,'voltage_a':V,'voltage_b':V,'return':V})
def series_voltage_residual(total_voltage: NDArray[np.float64],voltage_a: NDArray[np.float64],voltage_b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return total voltage minus two voltage drops, in volts.

    Inputs are finite real, broadcastable volts. Return float64 without mutation;
    reject invalid shapes or floating-point errors. Does not establish topology.
    """
    total,a,b=_arrays(total_voltage,voltage_a,voltage_b)
    with np.errstate(all='raise'):return _result(total-a-b)


@register_atom(witness_resistance,dim_map={'total_resistance':R,'resistance_a':R,'resistance_b':R,'return':R})
def series_resistance_residual(total_resistance: NDArray[np.float64],resistance_a: NDArray[np.float64],resistance_b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return total resistance minus two component resistances, in ohms.

    Inputs are finite real, broadcastable ohms. Return float64 without mutation;
    reject invalid shapes or floating-point errors. Does not establish topology.
    """
    total,a,b=_arrays(total_resistance,resistance_a,resistance_b)
    with np.errstate(all='raise'):return _result(total-a-b)


@register_atom(witness_balance,dim_map={'current':I,'total_resistance':R,'resistance_a':R,'resistance_b':R,'return':V})
def series_ohmic_balance_residual(current: NDArray[np.float64],total_resistance: NDArray[np.float64],resistance_a: NDArray[np.float64],resistance_b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return I*Rtotal - I*Ra - I*Rb, in volts, for a supplied common current.

    Inputs are finite real amperes and ohms with compatible broadcast shapes.
    Return float64 without mutation; reject floating-point errors. At zero current
    the residual is zero regardless of resistance, so it cannot validate Rtotal.
    """
    i,total,a,b=_arrays(current,total_resistance,resistance_a,resistance_b)
    with np.errstate(all='raise'):return _result(i*total-i*a-i*b)
