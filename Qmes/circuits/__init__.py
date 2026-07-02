"""Qmes/circuits/__init__.py

Quantum encoding circuit pool and kernel computation.
"""
from Qmes.circuits.registry import (
    CIRCUIT_POOL,
    UNIT_RANGE_CIRCUITS,
    get_circuit_names,
    get_circuit_fn,
    compute_kernel_matrix,
)

__all__ = [
    "CIRCUIT_POOL",
    "UNIT_RANGE_CIRCUITS",
    "get_circuit_names",
    "get_circuit_fn",
    "compute_kernel_matrix",
]