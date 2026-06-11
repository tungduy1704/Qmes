"""qmatch/extractors/clustering.py

Meta-feature extraction cho clustering (unsupervised).
KHÔNG cần y — hoàn toàn unsupervised.

Components:
    1. Hopkins statistic — cluster tendency
    2. Internal CVIs trên predicted clusters (k-means, range k) — NOT on true labels
    3. Distance distribution features (Ferrari & de Castro 2015 simplified)
    4. Intrinsic dimensionality (PCA-based + MLE)
    5. Simple meta-features

Input: X (n_samples, n_features), y=None
Output: fixed-length vector

Scaling: StandardScaler internally (needed for distance-based measures).

Dependencies: sklearn, scipy, pyclustertend (optional, fallback implemented)
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from Qmes.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

_FEATURE_NAMES: list[str] = [
    # Cluster tendency
    "hopkins",
    # Internal CVIs (best over k=2..10, on k-means predictions)
    "best_silhouette",      # best silhouette score
    "best_k_silhouette",    # k that gave best silhouette
    "best_ch",              # best Calinski-Harabasz
    "best_db",              # best (lowest) Davies-Bouldin
    # Distance distribution (simplified Ferrari & de Castro)
    "dist_mean",
    "dist_std",
    "dist_skewness",
    "dist_kurtosis",
    # Intrinsic dimensionality
    "pca_95_dim",           # #components for 95% variance
    "dim_eff",              # effective dimensionality
    # Simple
    "n_instances",
    "n_features",
    "attr_to_inst",
]


def _hopkins_statistic(X: np.ndarray, m: int | None = None) -> float:
    """Hopkins statistic for cluster tendency.

    H ≈ 0.5 → uniform (no clusters)
    H → 1.0 → strong cluster structure

    Convention chuẩn theo Hopkins 1954.
    Không dùng pyclustertend vì library đó dùng convention ngược (H → 0 là có cụm).
    """
    from sklearn.neighbors import NearestNeighbors

    n, d = X.shape
    if m is None:
        m = max(5, n // 10)
    m = min(m, n - 1)

    rng = np.random.RandomState(42)

    # w: distances từ sample points đến nearest neighbor trong X (bỏ self)
    # → nhỏ khi có cụm
    idx = rng.choice(n, size=m, replace=False)
    sample_points = X[idx]
    nn_real = NearestNeighbors(n_neighbors=2).fit(X)
    w_dists = nn_real.kneighbors(sample_points)[0][:, 1]  # col 0 là self (dist=0)

    # u: distances từ random points đến nearest real point
    # → lớn khi có cụm (random points rơi vào vùng trống)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    random_points = rng.uniform(mins, maxs, size=(m, d))
    nn_for_random = NearestNeighbors(n_neighbors=1).fit(X)
    u_dists = nn_for_random.kneighbors(random_points)[0][:, 0]

    u_sum = (u_dists ** d).sum()
    w_sum = (w_dists ** d).sum()

    if u_sum + w_sum < 1e-12:
        return 0.5
    return float(u_sum / (u_sum + w_sum))


def _compute_cvi_over_k(
    X: np.ndarray, k_range: range
) -> dict[str, float]:
    """Compute internal CVIs over a range of k values.

    Returns best scores (max silhouette, max CH, min DB) and best k.
    """
    best_sil = -1.0
    best_k_sil = 2
    best_ch = 0.0
    best_db = float("inf")

    for k in k_range:
        if k >= X.shape[0]:
            break
        try:
            km = KMeans(n_clusters=k, n_init=5, random_state=42, max_iter=100)
            labels = km.fit_predict(X)

            # Need at least 2 clusters assigned
            if len(set(labels)) < 2:
                continue

            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)

            if sil > best_sil:
                best_sil = sil
                best_k_sil = k
            if ch > best_ch:
                best_ch = ch
            if db < best_db:
                best_db = db
        except Exception:
            continue

    if best_db == float("inf"):
        best_db = 0.0

    return {
        "best_silhouette": best_sil,
        "best_k_silhouette": float(best_k_sil),
        "best_ch": best_ch,
        "best_db": best_db,
    }


def _distance_distribution_features(X: np.ndarray) -> dict[str, float]:
    """Simplified Ferrari & de Castro 2015: stats on pairwise distances.

    Subsample nếu n > 500 để tránh O(n²) quá lớn.
    """
    from scipy.spatial.distance import pdist
    from scipy.stats import kurtosis, skew

    n = X.shape[0]
    if n > 500:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=500, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    dists = pdist(X_sub, metric="euclidean")

    if len(dists) == 0:
        return {"dist_mean": 0.0, "dist_std": 0.0,
                "dist_skewness": 0.0, "dist_kurtosis": 0.0}

    return {
        "dist_mean": float(dists.mean()),
        "dist_std": float(dists.std()),
        "dist_skewness": float(skew(dists)),
        "dist_kurtosis": float(kurtosis(dists)),
    }


def _intrinsic_dim_features(X: np.ndarray) -> dict[str, float]:
    """PCA-based intrinsic dimensionality estimates."""
    from sklearn.decomposition import PCA

    # PCA 95% variance
    pca = PCA().fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    pca_95 = float(np.searchsorted(cumvar, 0.95) + 1)

    # Effective dimensionality
    eigenvalues = pca.explained_variance_
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if len(eigenvalues) == 0 or eigenvalues.sum() < 1e-12:
        dim_eff = 0.0
    else:
        total = eigenvalues.sum()
        dim_eff = float(total**2 / (eigenvalues**2).sum())

    return {"pca_95_dim": pca_95, "dim_eff": dim_eff}


class ClusteringExtractor(BaseExtractor):
    """Meta-feature extractor cho clustering (unsupervised).

    Produces 14-dim vector:
        [hopkins] + [4 CVI features] + [4 distance features]
        + [2 intrinsic dim] + [3 simple]

    Hoàn toàn unsupervised — không cần y.
    Internal CVIs tính trên k-means predictions (k=2..10), KHÔNG trên true labels.
    """

    def __init__(self, k_min: int = 2, k_max: int = 10):
        self.k_range = range(k_min, k_max + 1)

    @property
    def task_type(self) -> str:
        return "clustering"

    @property
    def _feature_names(self) -> list[str]:
        return _FEATURE_NAMES

    def _extract_raw(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> np.ndarray:
        # y is ignored — clustering is unsupervised

        # Scale internally for distance-based measures
        X_scaled = StandardScaler().fit_transform(X)

        # 1. Hopkins
        hopkins = _hopkins_statistic(X_scaled)

        # 2. Internal CVIs over k range
        cvi = _compute_cvi_over_k(X_scaled, self.k_range)

        # 3. Distance distribution
        dist = _distance_distribution_features(X_scaled)

        # 4. Intrinsic dimensionality
        idim = _intrinsic_dim_features(X_scaled)

        # 5. Simple
        n_inst = float(X.shape[0])
        n_feat = float(X.shape[1])
        attr_to_inst = n_feat / n_inst if n_inst > 0 else 0.0

        return np.array([
            hopkins,
            cvi["best_silhouette"], cvi["best_k_silhouette"],
            cvi["best_ch"], cvi["best_db"],
            dist["dist_mean"], dist["dist_std"],
            dist["dist_skewness"], dist["dist_kurtosis"],
            idim["pca_95_dim"], idim["dim_eff"],
            n_inst, n_feat, attr_to_inst,
        ], dtype=np.float64)