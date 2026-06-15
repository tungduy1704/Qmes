"""Qmes/extractors/classification.py

Meta-feature extraction for tabular classification.
Uses problexity (22 Lorena complexity measures) + 2 custom metrics from Qsun.

Input : X (n_samples, n_features), y (n_samples,) — binary or multiclass
Output: 24-dim vector

Scaling: problexity requires data in [0, 1]. Extractor handles scaling internally.

Dependencies: problexity, sklearn
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from Qmes.extractors.base import BaseExtractor

# ── 22 Lorena measures ────────────────────────────────
_PROBLEXITY_NAMES: list[str] = [
    # Feature-based (F)
    "f1", "f1v", "f2", "f3", "f4",
    # Linearity (L)
    "l1", "l2", "l3",
    # Neighborhood (N)
    "n1", "n2", "n3", "n4",
    # Network
    "t1", "lsc",
    "density", "cls_coef", "hubs",
    # Dimensionality
    "t2", "t3", "t4",
    # Class balance
    "c1", "c2",
]

_PX_EXPECTED_KEYS: list[str] = [
    "clsCoef" if n == "cls_coef" else n for n in _PROBLEXITY_NAMES
]

# 2 custom metrics (from Qsun/Qdata.py)
_CUSTOM_NAMES: list[str] = ["dim_eff", "kolmogorov"]

_ALL_NAMES: list[str] = _PROBLEXITY_NAMES
_STOCHASTIC_NAMES = ("l3", "n4") 
_STOCHASTIC_IDX = tuple(_PROBLEXITY_NAMES.index(nm) for nm in _STOCHASTIC_NAMES)
_K_SEEDS = 10


_STOCHASTIC_IDX = (7, 11)   # l3, n4: stochastic metrics excluded for determinism
_K_SEEDS = 10

def _compute_problexity(X: np.ndarray, y: np.ndarray, k_seeds: int = _K_SEEDS) -> np.ndarray:
    """22 Lorena complexity measures; l3/n4 (stochastic) averaged over k_seeds for stability."""
    import problexity as px
    runs = []
    for s in range(k_seeds):
        np.random.seed(s)
        cc = px.ComplexityCalculator(); cc.fit(X, y)
        if s == 0 and list(cc.report()["complexities"].keys()) != _PX_EXPECTED_KEYS:
            raise RuntimeError("problexity metric order đã đổi — positional labeling sai")
        runs.append(np.asarray(cc.complexity, dtype=np.float64))
    runs = np.vstack(runs)            # (K, 22)
    out = runs[0].copy()              # 20 deterministic features — identical across all seeds
    for j in _STOCHASTIC_IDX:
        out[j] = runs[:, j].mean()    # average l3, n4 separately across seeds
    return out

class ClassificationExtractor(BaseExtractor):
    """Meta-feature extractor for tabular classification.

    Produces 22-dim vector:
        [22 Lorena measures]
    """

    @property
    def task_type(self) -> str:
        return "classification"

    @property
    def _feature_names(self) -> list[str]:
        return _ALL_NAMES

    def _extract_raw(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> np.ndarray:
        if y is None:
            raise ValueError("Classification extractor requires y")

        # Scale to [0, 1] for problexity
        X_scaled = MinMaxScaler().fit_transform(X)

        # 22 Lorena measures
        complexity = _compute_problexity(X_scaled, y)

        return complexity