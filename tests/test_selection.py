"""Tests for Qmes.recommender.selection — MI feature selection + LOO evaluation.

This module is where the offline "best config" (the LOO heatmap) is decided,
so a bug here silently propagates into the shipped recommender. Two things get
special attention:

  1. The structural invariant  Single <= Tied <= Top3_Tied  (must hold by
     construction on every row — see the proof in the test docstring).
  2. The POSITIONAL alignment contract: run_loo_evaluation pairs row i of
     `meta_features` with `pivot_scores.columns[i]` purely by position. There is
     no name-based join, so if a caller builds meta-features in a different
     dataset order than the pivot, every label is silently mismatched. The
     alignment test below turns that silent corruption into a loud failure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from Qmes.recommender.selection import (
    DEFAULT_CLASSIFIERS,
    select_features_mi,
    run_loo_evaluation,
)


# ── helpers ───────────────────────────────────────────────────────────────
def _perfect_meta_pivot(n_per_group: int = 4):
    """Build a meta-dataset where meta-feature 0 PERFECTLY determines the
    best circuit (group g -> circuit[g] scores 1.0, others 0.0).

    Returns (meta_features ndarray, pivot DataFrame index=circuits cols=ds).
    Row i of meta aligns with column i of pivot.
    """
    circuits = ["A", "B", "C"]
    meta, ds_names = [], []
    scores = {c: [] for c in circuits}
    idx = 0
    for g in range(len(circuits)):
        for _ in range(n_per_group):
            ds_names.append(f"d{idx}")
            meta.append([float(g), 0.0, float(g) * 0.1])  # feature 0 = signal
            best = circuits[g]
            for c in circuits:
                scores[c].append(1.0 if c == best else 0.0)
            idx += 1
    meta = np.asarray(meta, dtype=float)
    pivot = pd.DataFrame(scores, index=ds_names).T   # index=circuits, cols=ds
    pivot.index.name = "circuit"
    return meta, pivot, circuits


_FAST_POOL = {
    "DT": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=3),
}


# ── classifier pool ───────────────────────────────────────────────────────
def test_default_pool_has_14_estimators():
    assert len(DEFAULT_CLASSIFIERS) == 14
    for name, est in DEFAULT_CLASSIFIERS.items():
        assert hasattr(est, "fit") and hasattr(est, "predict"), name


# ── MI feature selection ──────────────────────────────────────────────────
def test_select_features_mi_full_is_all_indices():
    X = np.random.RandomState(0).rand(20, 8)
    y = np.array([0, 1] * 10)
    subsets = select_features_mi(X, y)
    assert subsets["full"] == list(range(8))


def test_select_features_mi_skips_k_above_dim():
    """d=8 -> top5 exists, top10/15/20 are skipped (k > d)."""
    X = np.random.RandomState(0).rand(20, 8)
    y = np.array([0, 1] * 10)
    subsets = select_features_mi(X, y)
    assert "top5" in subsets and len(subsets["top5"]) == 5
    for k in (10, 15, 20):
        assert f"top{k}" not in subsets


def test_select_features_mi_indices_valid_and_unique():
    X = np.random.RandomState(1).rand(30, 12)
    y = np.array([0, 1, 2] * 10)
    subsets = select_features_mi(X, y)
    for label, idx in subsets.items():
        assert len(set(idx)) == len(idx), f"{label} has duplicate indices"
        assert all(0 <= i < 12 for i in idx), f"{label} has out-of-range index"


def test_select_features_mi_deterministic():
    X = np.random.RandomState(2).rand(25, 10)
    y = np.array([0, 1] * 12 + [0])
    a = select_features_mi(X, y, random_state=7)
    b = select_features_mi(X, y, random_state=7)
    assert a == b


# ── LOO output schema + ranges ────────────────────────────────────────────
def test_loo_output_schema():
    meta, pivot, _ = _perfect_meta_pivot()
    subsets = {"full": [0, 1, 2]}
    df = run_loo_evaluation(meta, pivot, classifiers=_FAST_POOL,
                            feature_subsets=subsets, verbose=False)
    assert list(df.columns) == [
        "Features", "Classifier", "Single", "Tied", "Top3_Tied", "Mean_Regret"
    ]
    # one row per (subset x classifier)
    assert len(df) == len(subsets) * len(_FAST_POOL)


def test_loo_metric_ranges():
    meta, pivot, _ = _perfect_meta_pivot()
    df = run_loo_evaluation(meta, pivot, classifiers=_FAST_POOL,
                            feature_subsets={"full": [0, 1, 2]}, verbose=False)
    for col in ("Single", "Tied", "Top3_Tied"):
        assert df[col].between(0.0, 1.0).all(), col
    assert (df["Mean_Regret"] >= 0).all()


def test_loo_structural_invariant_single_le_tied_le_top3():
    """Single <= Tied <= Top3_Tied must hold on EVERY row.

    Proof: the argmax circuit (y_single) is always inside the tied set, and
    top1 is always inside top3. So a correct Single implies a correct Tied,
    and a correct Tied implies a correct Top3_Tied. Using a noise-only
    meta-dataset makes the metrics non-trivial (not all 1.0), so this test
    isn't vacuously satisfied.
    """
    rng = np.random.RandomState(0)
    _, pivot, _ = _perfect_meta_pivot()
    noise_meta = rng.rand(pivot.shape[1], 3)   # meta carries no signal
    df = run_loo_evaluation(noise_meta, pivot, classifiers=_FAST_POOL,
                            feature_subsets={"full": [0, 1, 2]}, verbose=False)
    assert (df["Single"] <= df["Tied"] + 1e-9).all()
    assert (df["Tied"] <= df["Top3_Tied"] + 1e-9).all()


# ── behavioral: perfect signal must be recovered ──────────────────────────
def test_loo_perfect_signal_zero_regret():
    meta, pivot, _ = _perfect_meta_pivot()
    df = run_loo_evaluation(
        meta, pivot,
        classifiers={"DT": DecisionTreeClassifier(random_state=42)},
        feature_subsets={"full": [0, 1, 2]}, verbose=False,
    )
    row = df.iloc[0]
    assert row["Single"] == 1.0
    assert row["Tied"] == 1.0
    assert row["Mean_Regret"] == 0.0


# ── the silent-corruption guard: positional alignment ─────────────────────
def test_loo_requires_positional_alignment():
    """run_loo_evaluation aligns meta-row i with pivot-column i BY POSITION.

    With correct alignment + perfect signal, regret is 0. If the meta rows are
    permuted relative to the pivot columns (the exact failure mode when a
    cached loader order diverges from a fresh-build order), the same data must
    now produce strictly worse results — proving the function depends on, and
    does not self-check, alignment. This is the regression guard for that bug.
    """
    meta, pivot, _ = _perfect_meta_pivot()
    clf = {"DT": DecisionTreeClassifier(random_state=42)}

    aligned = run_loo_evaluation(meta, pivot, classifiers=clf,
                                 feature_subsets={"full": [0, 1, 2]},
                                 verbose=False)
    aligned_regret = aligned.iloc[0]["Mean_Regret"]

    # Permute meta rows so they no longer match pivot columns (mixes groups).
    perm = np.array([7, 2, 11, 0, 5, 9, 1, 6, 10, 3, 8, 4])
    misaligned = run_loo_evaluation(meta[perm], pivot, classifiers=clf,
                                    feature_subsets={"full": [0, 1, 2]},
                                    verbose=False)
    misaligned_regret = misaligned.iloc[0]["Mean_Regret"]

    assert aligned_regret == 0.0
    assert misaligned_regret > aligned_regret


# ── alignment guard (dataset_names) ───────────────────────────────────────
def test_loo_dataset_names_matching_passes():
    meta, pivot, _ = _perfect_meta_pivot()
    df = run_loo_evaluation(
        meta, pivot, classifiers=_FAST_POOL,
        feature_subsets={"full": [0, 1, 2]}, verbose=False,
        dataset_names=list(pivot.columns),
    )
    assert len(df) == len(_FAST_POOL)


def test_loo_dataset_names_mismatch_raises():
    meta, pivot, _ = _perfect_meta_pivot()
    bad = list(pivot.columns)
    bad[0], bad[1] = bad[1], bad[0]   # swap two names -> misaligned
    with pytest.raises(ValueError, match="alignment"):
        run_loo_evaluation(
            meta, pivot, classifiers=_FAST_POOL,
            feature_subsets={"full": [0, 1, 2]}, verbose=False,
            dataset_names=bad,
        )


def test_loo_wrong_row_count_raises():
    """Always-on shape guard (no dataset_names needed)."""
    meta, pivot, _ = _perfect_meta_pivot()
    with pytest.raises(ValueError, match="rows but pivot"):
        run_loo_evaluation(
            meta[:-1], pivot, classifiers=_FAST_POOL,
            feature_subsets={"full": [0, 1, 2]}, verbose=False,
        )


# ── default feature_subsets path (None -> MI internally) ──────────────────
def test_loo_default_subsets_runs():
    """When feature_subsets is None, it is derived via MI from the pivot's
    per-dataset best circuit. d=3 here, so only 'full' survives."""
    meta, pivot, _ = _perfect_meta_pivot()
    df = run_loo_evaluation(meta, pivot, classifiers=_FAST_POOL, verbose=False)
    labels = set(df["Features"])
    assert "full" in labels
    assert all(lab == "full" for lab in labels)  # top5+ skipped at d=3