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
    # Match the exact guard: if the task-type check is disabled, the
    # feature-name check downstream also says "mismatch" and would mask it.
    with pytest.raises(ValueError, match="Task-type mismatch"):
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
        assert len(row["rec_top_k"]) == 3   # top_k=3 was requested
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
    """Extractor and recommender agree (classification) and only the
    evaluator is off — so evaluate_recommendation's own guard is the one
    that must fire, before any quantum evaluation starts."""
    X, y = tiny_clf
    with pytest.raises(ValueError, match="Task-type mismatch"):
        evaluate_recommendation(
            {"d0": (X, y)},
            extractor=get_extractor("classification"),
            recommender=load_default_recommender("classification"),
            evaluator=get_evaluator("regression"),
        )


# ── documented defaults of the public API ─────────────────────────────────
def test_recommend_top_k_default_and_override(tiny_clf):
    """top_k defaults to 3 and must honour an explicit override."""
    X, y = tiny_clf
    ext = get_extractor("classification")
    rec = load_default_recommender("classification")

    assert len(recommend(X, y, extractor=ext, recommender=rec)["top_k"]) == 3
    assert len(recommend(X, y, extractor=ext, recommender=rec, top_k=5)["top_k"]) == 5


def test_recommend_preprocesses_raw_pandas_by_default(tiny_clf):
    """recommend() promises preprocess=True by default: a DataFrame with a
    categorical column must be encoded, not passed raw to the extractor."""
    rng = np.random.RandomState(0)
    n = 60
    X = pd.DataFrame({
        "num1": rng.rand(n),
        "num2": rng.rand(n),
        "cat": rng.choice(["red", "green", "blue"], size=n),
    })
    y = pd.Series(rng.randint(0, 2, size=n).astype(float))

    out = recommend(
        X, y,
        extractor=get_extractor("classification"),
        recommender=load_default_recommender("classification"),
    )
    assert len(out["ranking"]) == 7


# ── subsampling contract of preprocess_new_dataset ────────────────────────
def test_preprocess_subsample_is_deterministic():
    """Fixed random_state -> identical subsample on repeated calls, in both
    the random and the stratified branch."""
    rng = np.random.RandomState(0)
    X = rng.rand(150, 3)
    y = (rng.rand(150) > 0.5).astype(float)

    for stratify in (False, True):
        X1, y1 = preprocess_new_dataset(X, y, max_samples=100, stratify=stratify)
        X2, y2 = preprocess_new_dataset(X, y, max_samples=100, stratify=stratify)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


def test_preprocess_subsamples_without_replacement():
    """No row may be picked twice: with all-unique input rows, the
    subsample must contain max_samples distinct rows."""
    rng = np.random.RandomState(0)
    X = rng.rand(150, 3)          # unique rows almost surely
    y = rng.rand(150)

    X_out, _ = preprocess_new_dataset(X, y, max_samples=100)
    assert len(np.unique(X_out, axis=0)) == 100


def test_preprocess_at_exact_cap_is_passthrough(tiny_clf):
    """n == max_samples is NOT 'too large': data must pass through
    untouched (no shuffling, no resampling)."""
    X, y = tiny_clf   # n=60
    X_out, y_out = preprocess_new_dataset(X, y, max_samples=60)
    np.testing.assert_array_equal(X_out, X)
    np.testing.assert_array_equal(y_out, y)


# ── failure handling in evaluate_recommendation (stub Oracle) ─────────────
class _StubEvaluator:
    """Oracle stand-in returning a scripted score table per call.

    Lets tests exercise evaluate_recommendation's failure handling
    (all-NaN datasets) and scoring bookkeeping deterministically, without
    quantum evaluation and without needing the real Oracle to fail.
    """

    task_type = "classification"
    metric_name = "STUB"

    def __init__(self, per_call_scores):
        self._per_call = list(per_call_scores)
        self._calls = 0

    def evaluate_all(self, X, y):
        scores = self._per_call[self._calls]
        self._calls += 1
        return {c: {"mean_stub": s} for c, s in scores.items()}


def test_evaluate_recommendation_skips_failed_dataset(tiny_clf, circuit_names):
    """A dataset where every circuit fails must be skipped — and must NOT
    abort evaluation of the datasets after it."""
    X, y = tiny_clf
    all_nan = {c: np.nan for c in circuit_names}
    good = {c: 0.5 for c in circuit_names}
    good["RY"] = 0.9    # unique, clearly-best circuit
    all_tied = {c: 0.7 for c in circuit_names}   # every circuit tied-best

    df = evaluate_recommendation(
        {"bad": (X, y), "good": (X[:40], y[:40]), "tied": (X[:50], y[:50])},
        extractor=get_extractor("classification"),
        recommender=load_default_recommender("classification"),
        evaluator=_StubEvaluator([all_nan, good, all_tied]),
    )

    assert list(df["dataset"]) == ["good", "tied"]   # 'bad' skipped only
    row = df.loc[df["dataset"] == "good"].iloc[0]
    assert len(row["rec_top_k"]) == 3   # default top_k
    assert row["true_best"] == "RY"
    assert row["best_score"] == 0.9
    # scores are scripted, so every derived field is checkable exactly
    expected_rec_score = good[row["rec_top1"]]
    assert row["rec_score"] == pytest.approx(expected_rec_score)
    assert row["regret"] == pytest.approx(0.9 - expected_rec_score)
    tied = {c for c, s in good.items() if s >= 0.9 - 0.01}   # == {"RY"}
    assert row["tied_hit"] == (row["rec_top1"] in tied)
    assert row["top3_tied_hit"] == any(c in tied for c in row["rec_top_k"])

    # all circuits tied-best: any recommendation whatsoever is a tied hit
    row = df.loc[df["dataset"] == "tied"].iloc[0]
    assert row["regret"] == 0
    assert row["tied_hit"]
    assert row["top3_tied_hit"]


def test_evaluate_recommendation_empty_input_returns_empty_df():
    df = evaluate_recommendation(
        {},
        extractor=get_extractor("classification"),
        recommender=load_default_recommender("classification"),
        evaluator=_StubEvaluator([]),
    )
    assert len(df) == 0


# ── tied-set boundary + argument propagation ──────────────────────────────
def test_tied_set_includes_exact_boundary_score(tiny_clf, circuit_names):
    """The tied-best set is defined by s >= best - tied_threshold: a circuit
    landing EXACTLY on the boundary is tied. Scores are binary-exact
    (0.75 - 0.25 == 0.5 in floating point) so the comparison really sits on
    the boundary rather than a float hair to either side."""
    X, y = tiny_clf
    ext = get_extractor("classification")
    rec = load_default_recommender("classification")

    # The recommendation for a fixed dataset is deterministic - probe it,
    # then script the oracle so the recommended circuit sits exactly on the
    # boundary while a different circuit is strictly best.
    rec_top1 = recommend(X, y, extractor=ext, recommender=rec)["top_k"][0]
    best = next(c for c in circuit_names if c != rec_top1)
    scores = {c: 0.1 for c in circuit_names}
    scores[rec_top1] = 0.5          # == best - tied_threshold, exactly
    scores[best] = 0.75

    df = evaluate_recommendation(
        {"d0": (X, y)},
        extractor=ext, recommender=rec,
        evaluator=_StubEvaluator([scores]),
        tied_threshold=0.25,
    )
    row = df.iloc[0]
    assert row["regret"] == pytest.approx(0.25)
    assert row["tied_hit"], "boundary circuit must count as tied (>=, not >)"


def test_evaluate_recommendation_propagates_top_k(tiny_clf, circuit_names):
    """top_k must actually reach the recommender: with top_k=2 the row's
    rec_top_k has exactly 2 entries. 2 != the default of 3, so a silently
    dropped argument is visible."""
    X, y = tiny_clf
    good = {c: 0.5 for c in circuit_names}
    good["RY"] = 0.9
    df = evaluate_recommendation(
        {"d0": (X, y)},
        extractor=get_extractor("classification"),
        recommender=load_default_recommender("classification"),
        evaluator=_StubEvaluator([good]),
        top_k=2,
    )
    assert len(df.iloc[0]["rec_top_k"]) == 2


def test_evaluate_recommendation_preprocesses_by_default(circuit_names):
    """preprocess=True is the documented default: raw pandas input with a
    categorical column must be encoded before extraction, exactly as in
    recommend()."""
    rng = np.random.RandomState(0)
    n = 60
    X = pd.DataFrame({
        "num1": rng.rand(n),
        "num2": rng.rand(n),
        "cat": rng.choice(["red", "green", "blue"], size=n),
    })
    y = pd.Series(rng.randint(0, 2, size=n).astype(float))
    good = {c: 0.5 for c in circuit_names}
    good["RY"] = 0.9

    df = evaluate_recommendation(
        {"raw": (X, y)},
        extractor=get_extractor("classification"),
        recommender=load_default_recommender("classification"),
        evaluator=_StubEvaluator([good]),
    )
    assert list(df["dataset"]) == ["raw"]


def test_recommended_circuit_with_missing_metric_gives_nan_regret(
    tiny_clf, circuit_names
):
    """If the oracle returns no usable score for the recommended circuit,
    that circuit is dropped from the valid set and the row must report NaN
    regret/rec_score - not crash, and not fabricate a number - with
    tied_hit False."""
    X, y = tiny_clf
    ext = get_extractor("classification")
    rec = load_default_recommender("classification")
    rec_top1 = recommend(X, y, extractor=ext, recommender=rec)["top_k"][0]

    best = next(c for c in circuit_names if c != rec_top1)
    scores = {c: {"mean_stub": 0.5} for c in circuit_names}
    scores[best] = {"mean_stub": 0.9}
    scores[rec_top1] = {}            # metric key absent -> treated as failed

    class _RawStub(_StubEvaluator):
        def evaluate_all(self, X, y):
            return scores

    df = evaluate_recommendation(
        {"d0": (X, y)},
        extractor=ext, recommender=rec, evaluator=_RawStub([]),
    )
    row = df.iloc[0]
    assert np.isnan(row["regret"]) and np.isnan(row["rec_score"])
    assert not row["tied_hit"]
    assert row["best_score"] == pytest.approx(0.9)