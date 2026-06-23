"""End-to-end integration tests for the package's headline use case:

    pip install Qmes -> load a bundled recommender -> recommend a circuit
    for a brand-new dataset, with NO quantum evaluation at inference time.

If these fail after a fresh clone/install, the package's core promise is
broken regardless of how the unit tests look.
"""
from __future__ import annotations

import numpy as np
import pytest

from Qmes import get_extractor, load_default_recommender, recommend


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