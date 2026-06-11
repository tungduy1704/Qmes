from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from Qmes.extractors.base import BaseExtractor

_PROBLEXITY_NAMES: list[str] = [
    # Correlation (C1-C4)
    "c1", "c2", "c3", "c4",
    # Linearity (L1-L3)
    "l1", "l2", "l3",
    # Smoothness (S1-S4)
    "s1", "s2", "s3", "s4",
    # Dimensionality
    "t2",
]

_CUSTOM_NAMES: list[str] = ["dim_eff", "kolmogorov"]

_ALL_NAMES: list[str] = _PROBLEXITY_NAMES


def _compute_problexity_regression(
    X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    import problexity as px

    cc = px.ComplexityCalculator(mode="regression")
    cc.fit(X, y)
    return np.array(cc.complexity, dtype=np.float64)

class RegressionExtractor(BaseExtractor):
    @property
    def task_type(self) -> str:
        return "regression"

    @property
    def _feature_names(self) -> list[str]:
        return _ALL_NAMES

    def _extract_raw(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> np.ndarray:
        if y is None:
            raise ValueError("Regression extractor requires y")

        # Scale X to [0, 1]
        X_scaled = MinMaxScaler().fit_transform(X)

        # Scale y to [0, 1]
        y_min, y_max = y.min(), y.max()
        if y_max - y_min < 1e-12:
            y_scaled = np.zeros_like(y)
        else:
            y_scaled = (y - y_min) / (y_max - y_min)

        # 12 regression complexity measures
        complexity = _compute_problexity_regression(X_scaled, y_scaled)

        return complexity