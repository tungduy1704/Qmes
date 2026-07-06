"""Qmes/circuits/registry.py

Circuit pool and kernel matrix computation.
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
"""Registry of the seven encoding circuits shipped with Qmes.

Maps circuit name to an encoding function ``x -> quantum state``, where
``x`` is a single preprocessed sample (one rotation angle per qubit,
except ``HD`` which encodes three angles per qubit). All circuits are
implemented in the bundled Qsun simulator.

The pool is a plain dict and intentionally extensible: registering a new
circuit is adding an entry (``CIRCUIT_POOL["name"] = fn``) — no
subclassing required. If the new circuit expects inputs in ``[0, 1]``
rather than rotation angles in ``[0, pi]``, also add its name to
``UNIT_RANGE_CIRCUITS``.
"""

UNIT_RANGE_CIRCUITS = {"unit"}
"""Names of circuits whose inputs must be scaled to ``[0, 1]`` instead of ``[0, pi]``.

The evaluators pick the ``MinMaxScaler`` feature range per circuit from
this set. Only ``unit`` needs ``[0, 1]`` out of the box: its amplitude
encoding computes ``sqrt(x)`` and ``sqrt(1 - x)``, which is undefined
outside that interval. Getting this wrong for a custom circuit does not
raise — it silently degrades the kernel (see the Advanced Usage guide).
"""

def get_circuit_names() -> list[str]:
    """Return the names of all circuits in the pool.

    Returns
    -------
    list[str]
        Circuit names in the insertion order of ``CIRCUIT_POOL``:
    ``unit``, ``SRx``, ``RY``, ``HERx``, ``RY_CX``, ``ZFM``, ``HD``.
    """
    return list(CIRCUIT_POOL.keys())


def get_circuit_fn(name: str):
    """Look up an encoding function in ``CIRCUIT_POOL`` by name.

    Parameters
    ----------
    name : str
        Circuit name, e.g. ``'RY'``.

    Returns
    -------
    callable
        The encoding function ``x -> quantum state``.

    Raises
    ------
    ValueError
        If *name* is not in ``CIRCUIT_POOL``.
    """
    if name not in CIRCUIT_POOL:
        raise ValueError(
            f"Unknown circuit '{name}'. Available: {get_circuit_names()}"
        )
    return CIRCUIT_POOL[name]

# ── Kernel matrix ────────────────────────────────────────────────────────────
def compute_kernel_matrix(X1, X2, circuit_fn):
    """Compute the quantum fidelity kernel matrix between two sample sets.

    Each entry is the squared state overlap
    ``K[i, j] = |<phi(x1_i)|phi(x2_j)>|^2``, where ``phi`` is the feature
    map induced by *circuit_fn*. Every sample is encoded once
    (``n1 + n2`` circuit simulations), then all pairwise overlaps are
    taken — this is the O(n^2) cost that Qmes avoids at inference time.

    Parameters
    ----------
    X1 : ndarray, shape (n1, n_features)
        First sample set. Must already be scaled to the circuit's
        expected input range (see ``UNIT_RANGE_CIRCUITS``).
    X2 : ndarray, shape (n2, n_features)
        Second sample set. If *X2* is the same object as *X1*, the
        matrix is symmetric and only the upper triangle is computed.
    circuit_fn : callable
        Encoding function from ``CIRCUIT_POOL``.

    Returns
    -------
    ndarray, shape (n1, n2)
        Kernel matrix. Any NaN entries are replaced with 0.0 and logged
        as a warning rather than raised.
    """
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
