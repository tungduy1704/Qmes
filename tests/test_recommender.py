"""Tests for the PairwiseRecommender meta-learner.

Covers the contract that inference depends on:
  - fit on (meta-features, pivot) then predict -> ranking/top_k/votes
  - vote bookkeeping is internally consistent (sum == n_pairs)
  - the learnable synthetic signal is actually recovered (behavioral)
  - save/load round-trip preserves metadata AND predictions
  - feature_indices subsetting works
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from Qmes import get_recommender, PairwiseRecommender


def _fit(meta, pivot, **kw):
    rec = get_recommender(
        task_type="classification",
        classifier=DecisionTreeClassifier(random_state=0),
        feature_names=list(meta.columns),
        **kw,
    )
    return rec.fit(meta, pivot)


# ── factory wiring ────────────────────────────────────────────────────────
def test_get_recommender_fills_metric_name():
    rec = get_recommender("classification", DecisionTreeClassifier())
    assert rec.task_type == "classification"
    assert rec.metric_name == "MCC"
    rec_r = get_recommender("regression", DecisionTreeClassifier())
    assert rec_r.metric_name == "R2"


# ── predict structure ─────────────────────────────────────────────────────
def test_predict_structure(synthetic_meta_pivot):
    meta, pivot, circuits = synthetic_meta_pivot
    rec = _fit(meta, pivot)

    out = rec.predict(meta.iloc[0].values, top_k=3)
    assert set(out.keys()) == {"ranking", "top_k", "votes"}
    assert sorted(out["ranking"]) == sorted(circuits)   # permutation of all
    assert out["top_k"] == out["ranking"][:3]
    assert len(out["top_k"]) == 3
    # OvO: every pair casts exactly one vote
    n_pairs = len(circuits) * (len(circuits) - 1) // 2
    assert sum(out["votes"].values()) == n_pairs


def test_predict_recovers_learnable_signal(synthetic_meta_pivot):
    """group 0 -> 'A' best, group 1 -> 'B' best (see conftest)."""
    meta, pivot, _ = synthetic_meta_pivot
    rec = _fit(meta, pivot)

    q_group0 = np.array([0.0, 0.5, 0.5, 0.5])
    q_group1 = np.array([1.0, 0.5, 0.5, 0.5])
    assert rec.predict(q_group0)["top_k"][0] == "A"
    assert rec.predict(q_group1)["top_k"][0] == "B"


def test_predict_before_fit_raises(synthetic_meta_pivot):
    import pytest
    meta, _, _ = synthetic_meta_pivot
    rec = get_recommender("classification", DecisionTreeClassifier())
    with pytest.raises(RuntimeError, match="fit"):
        rec.predict(meta.iloc[0].values)


# ── save / load round-trip ────────────────────────────────────────────────
def test_save_load_roundtrip(synthetic_meta_pivot, tmp_path):
    meta, pivot, _ = synthetic_meta_pivot
    rec = _fit(meta, pivot)

    save_dir = tmp_path / "rec"
    rec.save(save_dir)
    loaded = PairwiseRecommender.load(save_dir)

    # metadata preserved
    assert loaded.task_type == rec.task_type
    assert loaded.metric_name == rec.metric_name
    assert list(loaded.feature_names) == list(rec.feature_names)
    assert loaded.circuits_ == rec.circuits_

    # predictions identical for several queries
    for i in range(len(meta)):
        q = meta.iloc[i].values
        assert loaded.predict(q) == rec.predict(q)


# ── feature subsetting ────────────────────────────────────────────────────
def test_feature_indices_subset(synthetic_meta_pivot):
    """Using only the informative column (index 0) must still recover the
    signal — and a fitted recommender restricts itself to those columns."""
    meta, pivot, _ = synthetic_meta_pivot
    rec = _fit(meta, pivot, feature_indices=[0])

    assert rec.predict(np.array([0.0, 9.9, 9.9, 9.9]))["top_k"][0] == "A"
    assert rec.predict(np.array([1.0, 9.9, 9.9, 9.9]))["top_k"][0] == "B"

def _tiny_fitted_recommender(classifier=None):
    """Minimal fitted recommender for format-v2 bundle tests."""
    rng = np.random.default_rng(0)
    circuits = ["unit", "SRx", "RY", "HERx", "RY_CX", "ZFM", "HD"]
    meta = rng.normal(size=(30, 12))
    pivot = pd.DataFrame(
        rng.uniform(size=(7, 30)),
        index=circuits, columns=[f"ds{i}" for i in range(30)],
    )
    rec = PairwiseRecommender(
        classifier or KNeighborsClassifier(),
        feature_indices=[0, 2, 4],
        task_type="regression", metric_name="R2",
        feature_names=[f"f{i}" for i in range(12)],
    ).fit(meta, pivot)
    return rec, rng.normal(size=12)


def test_save_writes_v2_bundle_and_roundtrip(tmp_path):
    rec, x = _tiny_fitted_recommender()
    rec.save(tmp_path)
    # Format v2: data files only, no pickled sklearn objects
    assert (tmp_path / "recommender.npz").is_file()
    assert (tmp_path / "recommender.json").is_file()
    assert not (tmp_path / "recommender.pkl").exists()
    rec2 = PairwiseRecommender.load(tmp_path)
    assert rec2.predict(x)["ranking"] == rec.predict(x)["ranking"]
    assert rec2.predict(x)["votes"] == rec.predict(x)["votes"]


def test_load_rejects_v1_pickle_bundle(tmp_path):
    (tmp_path / "recommender.pkl").write_bytes(b"")
    with pytest.raises(ValueError, match="format-v1"):
        PairwiseRecommender.load(tmp_path)


def test_save_rejects_non_json_serializable_classifier_params(tmp_path):
    # A callable metric cannot be stored in the JSON classifier spec;
    # save() must fail with TypeError, not write a half-broken bundle.
    rec, _ = _tiny_fitted_recommender(
        KNeighborsClassifier(metric=lambda a, b: float(np.sum(np.abs(a - b))))
    )
    with pytest.raises(TypeError):
        rec.save(tmp_path)