"""qmatch/extractors/base.py

Abstract base class cho tất cả task-specific extractors.
Mỗi extractor nhận raw data từ data loader, trả về fixed-length
meta-feature vector cùng danh sách tên feature tương ứng.

Output: (vector: np.ndarray shape (d,), feature_names: list[str])
    - d cố định cho mỗi task type, nhưng KHÁC NHAU giữa các task types
    - feature_names luôn có len == d
    - NaN/Inf trong vector được thay bằng 0.0 (fallback an toàn)

Scaling: mỗi concrete extractor tự lo scaling nội bộ trước khi tính.
Base class KHÔNG scale — vì mỗi thư viện cần convention khác nhau.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractionResult:
    """Container cho kết quả extraction.

    Attributes
    ----------
    vector : ndarray shape (d,), float64
        Meta-feature vector. Guaranteed: no NaN, no Inf.
    feature_names : list[str]
        Tên từng feature, len == len(vector).
    task_type : str
        Task type identifier (e.g. "classification", "timeseries").
    """

    vector: np.ndarray
    feature_names: list[str]
    task_type: str

    def __post_init__(self):
        if len(self.vector) != len(self.feature_names):
            raise ValueError(
                f"vector length ({len(self.vector)}) != "
                f"feature_names length ({len(self.feature_names)})"
            )

    def to_dict(self) -> dict[str, float]:
        """Return {feature_name: value} mapping."""
        return dict(zip(self.feature_names, self.vector.tolist()))

    @property
    def dim(self) -> int:
        return len(self.vector)


class BaseExtractor(ABC):
    """Abstract base class cho task-specific meta-feature extractors.

    Subclass cần implement:
        - task_type (property): str identifier
        - _feature_names (property): list tên features (cố định)
        - _extract_raw(X, y): tính meta-features, trả ndarray

    Lifecycle:
        1. User gọi extract(X, y) hoặc extract(X)
        2. Base class validate input
        3. Gọi _extract_raw(X, y) → raw vector
        4. Sanitize (NaN/Inf → 0.0)
        5. Wrap trong ExtractionResult
    """

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Task type identifier, e.g. 'classification'."""
        ...

    @property
    @abstractmethod
    def _feature_names(self) -> list[str]:
        """Fixed list of feature names. Length defines output dimension."""
        ...

    @abstractmethod
    def _extract_raw(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute raw meta-feature vector.

        Parameters
        ----------
        X : feature matrix or time series matrix
        y : target vector (None for unsupervised tasks like clustering)

        Returns
        -------
        ndarray shape (d,) where d == len(self._feature_names)
        """
        ...

    def extract(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> ExtractionResult:
        """Public API: extract meta-features from dataset.

        Parameters
        ----------
        X : ndarray
            - Tabular: (n_samples, n_features)
            - Time series: (n_samples, n_timesteps)
        y : ndarray or None
            - Classification/Regression/Anomaly: (n_samples,)
            - Multi-label: (n_samples, n_labels)
            - Clustering: None

        Returns
        -------
        ExtractionResult with sanitized vector + feature names
        """
        # ── Validate ────────────────────────────────────────────
        if X.ndim < 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] < 2:
            raise ValueError(f"Need at least 2 samples, got {X.shape[0]}")

        expected_dim = len(self._feature_names)

        # ── Extract ─────────────────────────────────────────────
        try:
            raw = self._extract_raw(X, y)
        except Exception:
            logger.exception("%s: extraction failed", self.task_type)
            raise

        # ── Sanitize ────────────────────────────────────────────
        raw = np.asarray(raw, dtype=np.float64).ravel()

        if len(raw) != expected_dim:
            logger.error(
                "%s: expected %d features, got %d. Padding/truncating.",
                self.task_type,
                expected_dim,
                len(raw),
            )
            fixed = np.zeros(expected_dim, dtype=np.float64)
            n = min(len(raw), expected_dim)
            fixed[:n] = raw[:n]
            raw = fixed

        bad_mask = ~np.isfinite(raw)
        if bad_mask.any():
            n_bad = bad_mask.sum()
            bad_names = [
                self._feature_names[i]
                for i in np.where(bad_mask)[0]
            ]
            logger.warning(
                "%s: %d non-finite values replaced with 0.0: %s",
                self.task_type,
                n_bad,
                bad_names[:5],  # log tối đa 5 tên
            )
            raw[bad_mask] = 0.0

        return ExtractionResult(
            vector=raw,
            feature_names=list(self._feature_names),
            task_type=self.task_type,
        )

    def extract_batch(
        self,
        datasets: dict[str, Union[np.ndarray, tuple[np.ndarray, ...]]],
    ) -> pd.DataFrame:
        """Extract meta-features cho tất cả datasets, trả DataFrame.

        Parameters
        ----------
        datasets : dict[name, data]
            Output từ data loader. Giá trị có thể là:
            - tuple (X, y) hoặc (X, Y) — supervised tasks
            - ndarray X — unsupervised tasks (clustering)

        Returns
        -------
        DataFrame shape (n_datasets, d)
            Index = dataset names, columns = feature names.
            Giống format meta-dataset trong paper:

            |                  | f1     | f1v    | ... | kolmogorov |
            |------------------|--------|--------|-----|------------|
            | Blobs_F2C2_S100  | 0.0068 | 0.0020 | ... | 0.9861     |
            | Iris_S80         | 0.0254 | 0.0079 | ... | 0.3643     |
        """
        rows: dict[str, list[float]] = {}
        skipped: list[str] = []

        for name, data in datasets.items():
            # Unpack: tuple → (X, y), ndarray → (X, None)
            if isinstance(data, tuple):
                X = data[0]
                y = data[1] if len(data) > 1 else None
            else:
                X = data
                y = None

            try:
                result = self.extract(X, y)
                rows[name] = result.vector.tolist()
            except Exception:
                logger.exception(
                    "%s: batch extraction failed for '%s', skipping",
                    self.task_type,
                    name,
                )
                skipped.append(name)

        if skipped:
            logger.warning(
                "%s: skipped %d/%d datasets: %s",
                self.task_type,
                len(skipped),
                len(datasets),
                skipped,
            )

        df = pd.DataFrame.from_dict(
            rows, orient="index", columns=self._feature_names
        )
        df.index.name = "dataset"
        return df