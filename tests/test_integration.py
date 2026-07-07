"""End-to-end integration tests for the package's headline use case:

    pip install Qmes -> load a bundled recommender -> recommend a circuit
    for a brand-new dataset, with NO quantum evaluation at inference time.

If these fail after a fresh clone/install, the package's core promise is
broken regardless of how the unit tests look.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Qmes import (
    evaluate_recommendation,
    get_evaluator,
    get_extractor,
    load_default_recommender,
    preprocess_new_dataset,
    recommend,
)


CASES = [
    ("classification", 22),
    ("regression", 12),
]


@pytest.mark.parametrize("task,expected_dim", CASES)
def test_recommend_end_to_end(task, expected_dim, tiny_clf, tiny_reg, circuit_names):
    X, y = tiny_clf if task == "classification" else tiny_reg

    extractor = get_extractor(task)
    recommender = load_default_recommender(task)
    out = recommend(X, y, extractor=extractor, recommender=recommender, top_k=3)

    # shape of the result
    assert set(out.keys()) == {"ranking", "top_k", "votes", "meta_features"}
    assert out["meta_features"].shape == (expected_dim,)

    # ranking is a permutation of the full circuit pool
    assert sorted(out["ranking"]) == sorted(circuit_names)

    # top-3 is valid: 3 distinct circuits, all from the pool
    assert len(out["top_k"]) == 3
    assert len(set(out["top_k"])) == 3
    assert all(c in circuit_names for c in out["top_k"])
    assert out["top_k"] == out["ranking"][:3]


def test_load_default_recommender_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task type"):
        load_default_recommender("anomaly-detection")


def test_recommend_is_deterministic(tiny_clf):
    """Same dataset + same bundled model -> stable recommendation.

    The *ranking* is the contract and must match exactly. The raw
    meta-feature vector is only guaranteed deterministic to floating-point
    tolerance: the `hubs` feature is an iterative igraph hub-score solve and
    can wobble by ~1 ULP on near-degenerate graphs (problexity itself warns
    the solution may be non-unique). That ULP never changes the ranking.
    """
    X, y = tiny_clf
    ext = get_extractor("classification")
    rec = load_default_recommender("classification")
    a = recommend(X, y, extractor=ext, recommender=rec)
    b = recommend(X, y, extractor=ext, recommender=rec)
    assert a["ranking"] == b["ranking"]
    np.testing.assert_allclose(a["meta_features"], b["meta_features"], atol=1e-9)


def test_task_type_mismatch_is_caught(tiny_clf):
    """A regression recommender with a classification extractor must error,
    not silently produce a meaningless recommendation."""
    X, y = tiny_clf
    clf_extractor = get_extractor("classification")
    reg_recommender = load_default_recommender("regression")
    with pytest.raises(ValueError, match="[Mm]ismatch"):
        recommend(X, y, extractor=clf_extractor, recommender=reg_recommender)


def test_feature_name_mismatch_is_caught(tiny_clf):
    """An extractor whose feature names differ from those the recommender
    was trained on must error — positional silent misalignment is exactly
    the defect class the LOO permutation test guards against."""
    X, y = tiny_clf
    ext = get_extractor("classification")
    rec = load_default_recommender("classification")
    rec.feature_names = ["definitely", "wrong", "names"]
    with pytest.raises(ValueError, match="Feature-name mismatch"):
        recommend(X, y, extractor=ext, recommender=rec)


# ── preprocess_new_dataset ────────────────────────────────────────────────
def test_preprocess_accepts_pandas_and_subsamples():
    """DataFrame/Series input with a categorical column and n > max_samples
    must come back numeric, float64, and capped at max_samples."""
    rng = np.random.RandomState(0)
    n = 150
    X = pd.DataFrame({
        "num": rng.rand(n),
        "cat": rng.choice(["red", "green", "blue"], size=n),
    })
    y = pd.Series(rng.randint(0, 2, size=n))

    X_out, y_out = preprocess_new_dataset(X, y, max_samples=100)

    assert isinstance(X_out, np.ndarray) and isinstance(y_out, np.ndarray)
    assert X_out.shape == (100, 2)
    assert y_out.shape == (100,)
    assert X_out.dtype == np.float64 and y_out.dtype == np.float64
    assert np.all(np.isfinite(X_out))


def test_preprocess_stratified_subsample_keeps_both_classes():
    """With a 90/10 imbalance, stratified subsampling must preserve the
    minority class at (approximately) its original ratio."""
    rng = np.random.RandomState(0)
    n = 200
    X = rng.rand(n, 3)
    y = np.array([0.0] * 180 + [1.0] * 20)

    X_out, y_out = preprocess_new_dataset(
        X, y, max_samples=100, stratify=True
    )

    assert len(X_out) == 100
    assert (y_out == 1.0).sum() == 10  # 10% of 100, exact under stratify


def test_preprocess_small_dataset_passes_through(tiny_clf):
    """n < max_samples -> no subsampling, data returned intact."""
    X, y = tiny_clf
    X_out, y_out = preprocess_new_dataset(X, y)
    np.testing.assert_array_equal(X_out, X)
    np.testing.assert_array_equal(y_out, y)


# ── evaluate_recommendation ───────────────────────────────────────────────
def test_evaluate_recommendation_end_to_end(tiny_clf, circuit_names):
    """Full inference evaluation on two datasets: recommend, run the real
    quantum Oracle, and check every reported column obeys its contract."""
    X, y = tiny_clf
    datasets = {"d0": (X, y), "d1": (X[:40], y[:40])}

    df = evaluate_recommendation(
        datasets,
        extractor=get_extractor("classification"),
        recommender=load_default_recommender("classification"),
        evaluator=get_evaluator("classification", n_splits=3),
        top_k=3,
    )

    assert list(df["dataset"]) == ["d0", "d1"]
    expected_cols = {
        "dataset", "rec_top1", "rec_top_k", "true_best", "true_top3",
        "tied_hit", "top3_tied_hit", "regret", "best_score", "rec_score",
    }
    assert set(df.columns) == expected_cols

    for _, row in df.iterrows():
        assert row["rec_top1"] in circuit_names
        assert row["true_best"] in circuit_names
        assert row["rec_top1"] == row["rec_top_k"][0]
        assert len(row["true_top3"]) == 3
        # regret is best - recommended, so non-negative and consistent
        assert row["regret"] >= 0
        assert row["regret"] == pytest.approx(
            row["best_score"] - row["rec_score"], abs=1e-4
        )
        # top1 hitting the tied set implies the top-k set hits it too
        if row["tied_hit"]:
            assert row["top3_tied_hit"]


def test_evaluate_recommendation_task_mismatch_raises(tiny_clf):
    X, y = tiny_clf
    with pytest.raises(ValueError, match="[Mm]ismatch"):
        evaluate_recommendation(
            {"d0": (X, y)},
            extractor=get_extractor("classification"),
            recommender=load_default_recommender("regression"),
            evaluator=get_evaluator("classification"),
        )