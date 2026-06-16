"""Qmes/circuits/registry.py

Circuit pool và kernel matrix computation.
"""

from __future__ import annotations
import numpy as np
from Qmes.Qsun.Qencodes import *
from Qmes.Qsun.Qkernels import *

# ── Circuit registry ─────────────────────────────────────────────────────────
CIRCUIT_POOL = {
    "unit": lambda x: unit_encode(x),
    "SRx":  lambda x: SeparableRXEncoding_encode(x),
    "RY":   lambda x: angle_encode(x),
    "HERx": lambda x: HardwareEfficientEmbeddingRx_encode(x),
    "RY_CX": lambda x: RY_CX_linear_encode(x),
    "ZFM":  lambda x: ZFeatureMap_encode(x),
    "HD":   lambda x: HighDim_encode(x),
}

# Circuits uses range [0, 1] instead of [0, π]
UNIT_RANGE_CIRCUITS = {"unit"}

def get_circuit_names() -> list[str]:
    return list(CIRCUIT_POOL.keys())


def get_circuit_fn(name: str):
    if name not in CIRCUIT_POOL:
        raise ValueError(
            f"Unknown circuit '{name}'. Available: {get_circuit_names()}"
        )
    return CIRCUIT_POOL[name]

# ── Kernel matrix ────────────────────────────────────────────────────────────

def compute_kernel_matrix(X1, X2, circuit_fn):
    states_1 = [circuit_fn(x) for x in X1]
    symmetric = X1 is X2
    states_2 = states_1 if symmetric else [circuit_fn(x) for x in X2]

    n1, n2 = len(states_1), len(states_2)
    K = np.zeros((n1, n2))
    if symmetric:
        for i in range(n1):
            K[i, i] = state_product(states_1[i], states_1[i]) ** 2
            for j in range(i + 1, n2):
                v = state_product(states_1[i], states_2[j]) ** 2
                K[i, j] = K[j, i] = v
    else:
        for i in range(n1):
            for j in range(n2):
                K[i, j] = state_product(states_1[i], states_2[j]) ** 2

    if np.isnan(K).any():
        n_nan = int(np.isnan(K).sum())
        import logging
        logging.getLogger(__name__).warning(
            "NaN in kernel matrix: %d entries replaced with 0.0", n_nan
        )
        K = np.nan_to_num(K, nan=0.0)

    return K
