"""Qmes/extractors/regression.py

Meta-feature extraction for tabular regression.
Uses problexity (mode='regression'): 12 complexity measures.

Input : X (n_samples, n_features), y (n_samples,) — continuous target
Output: 12-dim vector

Scaling: problexity requires data in [0, 1]. Extractor handles scaling of X and y internally.

Dependencies: problexity, sklearn
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from Qmes.extractors.base import BaseExtractor

_PROBLEXITY_NAMES: list[str] = [
    "c1", "c2", "c3", "c4",   # correlation
    "l1", "l2",               # linearity 1-2
    "s1", "s2", "s3",         # smoothness 1-3
    "l3",                     # linearity 3 
    "s4",                     # smoothness 4
    "t2",                     # dimensionality
]

_ALL_NAMES: list[str] = _PROBLEXITY_NAMES

_STOCHASTIC_NAMES = ("l3", "s4")
_STOCHASTIC_IDX = tuple(_PROBLEXITY_NAMES.index(nm) for nm in _STOCHASTIC_NAMES)
_K_SEEDS = 10


def _compute_problexity_regression(
    X: np.ndarray, y: np.ndarray, k_seeds: int = _K_SEEDS
) -> np.ndarray:
    """12 complexity measures; l3 and s4 (stochastic) averaged over k_seeds for stability."""
    import problexity as px

    runs = []
    for s in range(k_seeds):
        np.random.seed(s)
        cc = px.ComplexityCalculator(mode="regression")
        cc.fit(X, y)
        if s == 0:
            keys = list(cc.report()["complexities"].keys())
            if keys != _PROBLEXITY_NAMES:
                raise RuntimeError(
                    f"problexity regression metric order đã đổi:\n  {keys}\n"
                    f"!= {_PROBLEXITY_NAMES}\n  -> positional labeling sai"
                )
        runs.append(np.asarray(cc.complexity, dtype=np.float64))

    runs = np.vstack(runs)        # (K, 12)
    out = runs[0].copy()          
    for j in _STOCHASTIC_IDX:
        out[j] = runs[:, j].mean()  # average l3, s4
    return out


class RegressionExtractor(BaseExtractor):
    """Meta-feature extractor for tabular regression."""

    @property
    def task_type(self) -> str:
        return "regression"

    @property
    def _feature_names(self) -> list[str]:
        return _ALL_NAMES

    def _extract_raw(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        if y is None:
            raise ValueError("Regression extractor requires y")

        X_scaled = MinMaxScaler().fit_transform(X)

        y_min, y_max = y.min(), y.max()
        if y_max - y_min < 1e-12:
            y_scaled = np.zeros_like(y)
        else:
            y_scaled = (y - y_min) / (y_max - y_min)

        return _compute_problexity_regression(X_scaled, y_scaled)