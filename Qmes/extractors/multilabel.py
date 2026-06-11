"""qmatch/extractors/multilabel.py

Meta-feature extraction cho multi-label classification.
Dùng Charte et al. catalog: LCard, LDen, IRLbl, MeanIR, SCUMBLE, etc.

Input: X (n_samples, n_features), Y (n_samples, n_labels) — binary {0,1}
Output: fixed-length vector

Note: Y là 2D matrix, truyền vào vị trí tham số `y` của extract().

Scaling: không cần — metrics tính trên label matrix, không trên feature values.

Dependencies: numpy, scipy (cho SCUMBLE)
"""
from __future__ import annotations

import numpy as np

from Qmes.extractors.base import BaseExtractor

# ── Multi-label characterization metrics ────────────────────────────────────
# Charte & Charte 2015 (mldr), Tsoumakas et al. 2010, Read et al. 2011
_FEATURE_NAMES: list[str] = [
    # Volume metrics
    "n_instances",      # m
    "n_features",       # f
    "n_labels",         # q = |L|
    "f_over_q",         # f/q ratio
    # Label cardinality & density
    "LCard",            # (1/m) * Σ |Y_i|
    "LDen",             # LCard / q
    # Label diversity
    "LDiv",             # number of unique labelsets
    "PUniq",            # proportion of singleton labelsets
    # Imbalance metrics (Charte et al.)
    "MeanIR",           # mean imbalance ratio across labels
    "MaxIR",            # max imbalance ratio
    "CVIR",             # coefficient of variation of IRLbl
    # Concurrence & co-occurrence
    "MeanPL",           # mean proportion of labels per instance
    "MeanCL",           # mean co-occurrence between label pairs
    # SCUMBLE
    "SCUMBLE_mean",     # mean SCUMBLE score (concurrence-based)
    # Entropy
    "label_entropy",    # entropy of label frequency distribution
]


def _compute_multilabel_features(
    X: np.ndarray, Y: np.ndarray
) -> np.ndarray:
    """Compute multi-label meta-features from feature matrix X and label matrix Y.

    Parameters
    ----------
    X : (n_samples, n_features)
    Y : (n_samples, n_labels), binary {0, 1}
    """
    m, f = X.shape
    q = Y.shape[1]

    # ── Volume ──────────────────────────────────────────────────
    n_instances = float(m)
    n_features = float(f)
    n_labels = float(q)
    f_over_q = f / q if q > 0 else 0.0

    # ── Label cardinality & density ─────────────────────────────
    card_per_instance = Y.sum(axis=1)  # |Y_i| for each instance
    LCard = float(card_per_instance.mean())
    LDen = LCard / q if q > 0 else 0.0

    # ── Label diversity ─────────────────────────────────────────
    labelsets = set(tuple(row) for row in Y)
    LDiv = float(len(labelsets))
    # PUniq: proportion of labelsets that appear exactly once
    from collections import Counter
    labelset_counts = Counter(tuple(row) for row in Y)
    PUniq = sum(1 for c in labelset_counts.values() if c == 1) / max(LDiv, 1)

    # ── Imbalance ratio per label (IRLbl) ───────────────────────
    label_freqs = Y.sum(axis=0).astype(float)  # (q,)
    max_freq = label_freqs.max()
    # IRLbl_j = max_freq / freq_j (undefined if freq_j == 0)
    IRLbl = np.where(label_freqs > 0, max_freq / label_freqs, 0.0)
    MeanIR = float(IRLbl.mean()) if q > 0 else 0.0
    MaxIR = float(IRLbl.max()) if q > 0 else 0.0
    CVIR = float(IRLbl.std() / IRLbl.mean()) if IRLbl.mean() > 1e-12 else 0.0

    # ── Concurrence ─────────────────────────────────────────────
    MeanPL = LDen  # equivalent: mean proportion of labels per instance

    # Mean co-occurrence: fraction of instances sharing both labels
    if q > 1:
        co_occur_sum = 0.0
        n_pairs = 0
        for j in range(q):
            for k in range(j + 1, q):
                both = ((Y[:, j] == 1) & (Y[:, k] == 1)).sum()
                either = ((Y[:, j] == 1) | (Y[:, k] == 1)).sum()
                if either > 0:
                    co_occur_sum += both / either
                n_pairs += 1
        MeanCL = co_occur_sum / n_pairs if n_pairs > 0 else 0.0
    else:
        MeanCL = 0.0

    # ── SCUMBLE (Charte et al. 2019) ───────────────────────────
    # Per-instance concurrence measure
    # SCUMBLE_i = 1 - (1/(q-1)) * prod(IRLbl_j for active labels) ^ (1/|Y_i|)
    # Simplified version: uses IRLbl product ratio
    scumble_scores = np.zeros(m)
    for i in range(m):
        active = np.where(Y[i] == 1)[0]
        if len(active) <= 1:
            scumble_scores[i] = 0.0
            continue
        ir_active = IRLbl[active]
        mean_ir = ir_active.mean()
        if mean_ir < 1e-12:
            scumble_scores[i] = 0.0
            continue
        # SCUMBLE_i = 1 - (1/|Y_i|) * Σ_j (IRLbl_j / mean_IR_active)
        prod_ratio = np.prod(ir_active / mean_ir) ** (1.0 / len(active))
        scumble_scores[i] = 1.0 - prod_ratio

    SCUMBLE_mean = float(scumble_scores.mean())

    # ── Entropy ─────────────────────────────────────────────────
    # Entropy of label frequency distribution
    if q > 0 and label_freqs.sum() > 0:
        p = label_freqs / label_freqs.sum()
        p = p[p > 0]
        label_entropy = float(-np.sum(p * np.log2(p)))
    else:
        label_entropy = 0.0

    return np.array([
        n_instances, n_features, n_labels, f_over_q,
        LCard, LDen,
        LDiv, PUniq,
        MeanIR, MaxIR, CVIR,
        MeanPL, MeanCL,
        SCUMBLE_mean,
        label_entropy,
    ], dtype=np.float64)


class MultilabelExtractor(BaseExtractor):
    """Meta-feature extractor cho multi-label classification.

    Produces 15-dim vector from Charte/Tsoumakas catalog.

    Usage:
        ext = MultilabelExtractor()
        result = ext.extract(X, Y)  # Y is 2D binary matrix
    """

    @property
    def task_type(self) -> str:
        return "multilabel"

    @property
    def _feature_names(self) -> list[str]:
        return _FEATURE_NAMES

    def _extract_raw(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> np.ndarray:
        if y is None:
            raise ValueError("Multi-label extractor requires Y (label matrix)")
        if y.ndim != 2:
            raise ValueError(
                f"Multi-label Y must be 2D (n_samples, n_labels), got shape {y.shape}"
            )

        return _compute_multilabel_features(X, y)