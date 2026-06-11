"""qmatch/extractors/anomaly.py

Meta-feature extraction cho anomaly detection.
Dùng PyOD detector score statistics + dataset descriptors.

Components:
    1. LOF/iForest anomaly score statistics (computed unsupervised on X)
    2. Contamination rate (from y)
    3. Feature correlation statistics
    4. Intrinsic dimensionality
    5. Simple meta-features

Input: X (n_samples, n_features), y (n_samples,) binary {0=normal, 1=anomaly}
Output: fixed-length vector

Scaling: StandardScaler internally (PyOD detectors need it).

Dependencies: pyod, sklearn
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.preprocessing import StandardScaler

from Qmes.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

_FEATURE_NAMES: list[str] = [
    # Contamination
    "contamination_rate",
    # LOF score statistics (unsupervised)
    "lof_score_mean",
    "lof_score_std",
    "lof_score_skew",
    "lof_score_q90",       # 90th percentile
    # iForest score statistics (unsupervised)
    "iforest_score_mean",
    "iforest_score_std",
    "iforest_score_skew",
    "iforest_score_q90",
    # Feature correlation
    "mean_abs_corr",       # mean |correlation| between features
    "max_abs_corr",        # max |correlation|
    # Intrinsic dim
    "pca_95_dim",
    "dim_eff",
    # Simple
    "n_instances",
    "n_features",
    "n_normal",
    "n_anomaly",
]


def _score_statistics(scores: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute summary statistics of anomaly scores."""
    from scipy.stats import skew

    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_skew": 0.0,
            f"{prefix}_q90": 0.0,
        }
    return {
        f"{prefix}_mean": float(scores.mean()),
        f"{prefix}_std": float(scores.std()),
        f"{prefix}_skew": float(skew(scores)),
        f"{prefix}_q90": float(np.percentile(scores, 90)),
    }


def _compute_lof_scores(X: np.ndarray) -> np.ndarray:
    """Compute LOF anomaly scores (unsupervised)."""
    from pyod.models.lof import LOF

    clf = LOF(n_neighbors=min(20, X.shape[0] - 1))
    clf.fit(X)
    return clf.decision_scores_


def _compute_iforest_scores(X: np.ndarray) -> np.ndarray:
    """Compute Isolation Forest anomaly scores (unsupervised)."""
    from pyod.models.iforest import IForest

    clf = IForest(n_estimators=100, random_state=42)
    clf.fit(X)
    return clf.decision_scores_


def _correlation_features(X: np.ndarray) -> dict[str, float]:
    """Feature correlation statistics."""
    if X.shape[1] < 2:
        return {"mean_abs_corr": 0.0, "max_abs_corr": 0.0}

    corr = np.corrcoef(X, rowvar=False)
    # Extract upper triangle (exclude diagonal)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    upper = np.abs(corr[mask])
    upper = upper[np.isfinite(upper)]

    if len(upper) == 0:
        return {"mean_abs_corr": 0.0, "max_abs_corr": 0.0}

    return {
        "mean_abs_corr": float(upper.mean()),
        "max_abs_corr": float(upper.max()),
    }


def _intrinsic_dim_features(X: np.ndarray) -> dict[str, float]:
    """PCA-based intrinsic dimensionality."""
    from sklearn.decomposition import PCA

    pca = PCA().fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    pca_95 = float(np.searchsorted(cumvar, 0.95) + 1)

    eigenvalues = pca.explained_variance_
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if len(eigenvalues) == 0 or eigenvalues.sum() < 1e-12:
        dim_eff = 0.0
    else:
        total = eigenvalues.sum()
        dim_eff = float(total**2 / (eigenvalues**2).sum())

    return {"pca_95_dim": pca_95, "dim_eff": dim_eff}


class AnomalyExtractor(BaseExtractor):
    """Meta-feature extractor cho anomaly detection.

    Produces 17-dim vector:
        [contamination] + [4 LOF stats] + [4 iForest stats]
        + [2 correlation] + [2 intrinsic dim] + [4 simple]

    PyOD detectors chạy unsupervised trên X — scores mô tả
    mức độ "dễ detect" của anomalies trong dataset.
    y chỉ dùng để tính contamination rate.
    """

    @property
    def task_type(self) -> str:
        return "anomaly"

    @property
    def _feature_names(self) -> list[str]:
        return _FEATURE_NAMES

    def _extract_raw(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> np.ndarray:
        if y is None:
            raise ValueError("Anomaly detection extractor requires y")

        # Scale for PyOD
        X_scaled = StandardScaler().fit_transform(X)

        # Contamination
        n_anomaly = int((y == 1).sum())
        n_normal = int((y == 0).sum())
        n_total = X.shape[0]
        contamination = n_anomaly / n_total if n_total > 0 else 0.0

        # LOF scores
        try:
            lof_scores = _compute_lof_scores(X_scaled)
            lof_stats = _score_statistics(lof_scores, "lof_score")
        except Exception as e:
            logger.warning("LOF failed: %s", e)
            lof_stats = _score_statistics(np.array([]), "lof_score")

        # iForest scores
        try:
            if_scores = _compute_iforest_scores(X_scaled)
            if_stats = _score_statistics(if_scores, "iforest_score")
        except Exception as e:
            logger.warning("iForest failed: %s", e)
            if_stats = _score_statistics(np.array([]), "iforest_score")

        # Correlation
        corr = _correlation_features(X_scaled)

        # Intrinsic dim
        idim = _intrinsic_dim_features(X_scaled)

        return np.array([
            contamination,
            lof_stats["lof_score_mean"],
            lof_stats["lof_score_std"],
            lof_stats["lof_score_skew"],
            lof_stats["lof_score_q90"],
            if_stats["iforest_score_mean"],
            if_stats["iforest_score_std"],
            if_stats["iforest_score_skew"],
            if_stats["iforest_score_q90"],
            corr["mean_abs_corr"],
            corr["max_abs_corr"],
            idim["pca_95_dim"],
            idim["dim_eff"],
            float(n_total),
            float(X.shape[1]),
            float(n_normal),
            float(n_anomaly),
        ], dtype=np.float64)