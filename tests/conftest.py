"""Shared fixtures for the Qmes test suite.

Datasets are kept deliberately tiny so the quantum Oracle (O(n^2) circuit
simulations) stays fast enough to run in the default test session — no
`slow` marker needed at this size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from Qmes.circuits import get_circuit_names


# ── raw datasets ──────────────────────────────────────────────────────────
@pytest.fixture
def tiny_clf():
    """Small binary-classification dataset (n=60, d=4)."""
    X, y = make_classification(
        n_samples=60, n_features=4, n_informative=3, n_redundant=0,
        n_classes=2, random_state=0,
    )
    return X.astype(np.float64), y.astype(np.float64)


@pytest.fixture
def tiny_reg():
    """Small regression dataset (n=60, d=4)."""
    X, y = make_regression(
        n_samples=60, n_features=4, n_informative=3, noise=0.1,
        random_state=0,
    )
    return X.astype(np.float64), y.astype(np.float64)


# ── learnable synthetic meta-dataset for the recommender ──────────────────
@pytest.fixture
def synthetic_meta_pivot():
    """A small (meta-features, pivot) pair with a *learnable* signal.

    Construction: feature column 0 is a binary group indicator.
        group 0  -> circuit 'A' is strictly best
        group 1  -> circuit 'B' is strictly best
    A correctly-fitted recommender must therefore put 'A' on top for a
    group-0 query and 'B' on top for a group-1 query. This lets the tests
    assert *behavior*, not just output shape.

    Returns
    -------
    (meta, pivot, circuits)
        meta   : DataFrame (n_datasets, 4)
        pivot  : DataFrame index=circuits, columns=datasets
        circuits : list[str]
    """
    rng = np.random.RandomState(0)
    circuits = ["A", "B", "C", "D", "E"]
    n_per_group = 8
    n = 2 * n_per_group

    group = np.array([0] * n_per_group + [1] * n_per_group)
    meta = np.column_stack([
        group.astype(float),          # informative
        rng.rand(n),                  # noise
        rng.rand(n),                  # noise
        rng.rand(n),                  # noise
    ])
    ds_names = [f"d{i}" for i in range(n)]
    meta_df = pd.DataFrame(meta, index=ds_names, columns=["g", "x1", "x2", "x3"])

    # Build pivot: winner gets a clearly-higher score per dataset.
    scores = {}
    for i, ds in enumerate(ds_names):
        base = rng.rand(len(circuits)) * 0.3        # 0..0.3 for the losers
        col = dict(zip(circuits, base))
        winner = "A" if group[i] == 0 else "B"
        col[winner] = 0.9 + rng.rand() * 0.05       # clearly best
        scores[ds] = col
    pivot = pd.DataFrame(scores)        # index=circuits, columns=datasets
    pivot.index.name = "circuit"
    pivot = pivot.loc[circuits]         # enforce row order

    return meta_df, pivot, circuits


@pytest.fixture(scope="session")
def circuit_names():
    return get_circuit_names()