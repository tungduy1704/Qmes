"""qmatch/extractors/classification.py

Meta-feature extraction cho tabular classification.
Dùng problexity (22 Lorena complexity measures) + 2 custom metrics từ Qsun.

Input: X (n_samples, n_features), y (n_samples,) — binary hoặc multiclass
Output: 24-dim vector

Scaling: problexity cần data trong [0, 1]. Extractor tự scale nội bộ.

Dependencies: problexity, sklearn
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from Qmes.extractors.base import BaseExtractor

# ── 22 Lorena measures (problexity ordering) ────────────────────────────────
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

# 2 custom metrics (from Qsun/Qdata.py — prior paper)
_CUSTOM_NAMES: list[str] = ["dim_eff", "kolmogorov"]

_ALL_NAMES: list[str] = _PROBLEXITY_NAMES


def _compute_problexity(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute 22 Lorena complexity measures via problexity.

    X phải đã scale về [0, 1].
    """
    import problexity as px

    cc = px.ComplexityCalculator()
    cc.fit(X, y)
    # cc.complexity trả list 22 giá trị theo thứ tự chuẩn
    return np.array(cc.complexity, dtype=np.float64)

class ClassificationExtractor(BaseExtractor):
    """Meta-feature extractor cho tabular classification.

    Produces 24-dim vector:
        [22 Lorena measures] + [dim_eff, kolmogorov]

    Consistent với prior paper (200-dataset benchmark).
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