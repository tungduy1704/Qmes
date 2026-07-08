"""Tests for the PairwiseRecommender meta-learner.

Covers the contract that inference depends on:
  - fit on (meta-features, pivot) then predict -> ranking/top_k/votes
  - vote bookkeeping is internally consistent (sum == n_pairs)
  - the learnable synthetic signal is actually recovered (behavioral)
  - save/load round-trip preserves metadata AND predictions
  - feature_indices subsetting works
"""
from __future__ import annotations

import json

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

    # documented default: top_k=3 when not passed
    assert len(rec.predict(meta.iloc[0].values)["top_k"]) == 3


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


def test_fit_tie_break_favors_first_circuit(synthetic_meta_pivot):
    """On an EXACT score tie the pair label is 1 (>=), i.e. the pair vote
    goes to the first circuit of the pair. Pin the convention: with two
    identically-scoring circuits, the earlier one must never rank below
    the later one."""
    meta, pivot, _ = synthetic_meta_pivot
    tied = pivot.copy()
    tied.loc["B"] = tied.loc["A"]        # B is A's exact clone everywhere
    rec = _fit(meta, tied)

    for i in range(len(meta)):
        ranking = rec.predict(meta.iloc[i].values)["ranking"]
        assert ranking.index("A") < ranking.index("B")

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
        tied_threshold=0.05,      # non-default on purpose: round-trip must restore it
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


def test_save_creates_nested_directories(tmp_path):
    """save() promises to create the target directory, parents included."""
    rec, x = _tiny_fitted_recommender()
    nested = tmp_path / "models" / "v2" / "rec"
    rec.save(nested)
    loaded = PairwiseRecommender.load(nested)
    assert loaded.predict(x)["ranking"] == rec.predict(x)["ranking"]


def test_roundtrip_restores_full_configuration(tmp_path):
    """Every configuration field must survive save->load, not just the
    ones predict() happens to consume."""
    rec, _ = _tiny_fitted_recommender()
    # The constructor must store the argument itself, not fall back to a
    # default - otherwise the round-trip equality below compares defaults.
    assert rec.tied_threshold == 0.05
    rec.save(tmp_path)
    loaded = PairwiseRecommender.load(tmp_path)

    assert loaded.tied_threshold == rec.tied_threshold  # 0.05, non-default
    assert list(loaded.feature_indices) == list(rec.feature_indices)
    # training pivot reconstructed exactly: values + circuit index + dataset columns
    pd.testing.assert_frame_equal(loaded.train_pivot_, rec.train_pivot_)


def test_bundle_json_declares_format_version_2(tmp_path):
    rec, _ = _tiny_fitted_recommender()
    rec.save(tmp_path)
    meta = json.loads((tmp_path / "recommender.json").read_text())
    assert meta["format_version"] == 2


def test_load_missing_bundle_raises_file_not_found(tmp_path):
    """An empty directory is 'bundle missing' (FileNotFoundError), not a
    v1-format error (ValueError) — the two must stay distinguishable."""
    with pytest.raises(FileNotFoundError):
        PairwiseRecommender.load(tmp_path / "nothing_here")


def test_load_ignores_stray_pickle_next_to_v2_bundle(tmp_path):
    """A leftover recommender.pkl BESIDE a complete v2 bundle must not
    trigger the v1 rejection: that guard means 'v1-only directory' and is
    keyed on recommender.json being absent."""
    rec, x = _tiny_fitted_recommender()
    rec.save(tmp_path)
    (tmp_path / "recommender.pkl").write_bytes(b"")

    loaded = PairwiseRecommender.load(tmp_path)
    assert loaded.predict(x)["ranking"] == rec.predict(x)["ranking"]