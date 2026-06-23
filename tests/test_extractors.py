"""Unit tests for meta-feature extractors.

Key properties under test:
  - correct fixed output dimension per task (clf=22, reg=12)
  - determinism: stochastic Problexity metrics (l3/n4 for clf, l3/s4 for
    reg) are averaged over K seeds internally, so extract() must return the
    same vector on repeated calls AND must not depend on the caller's global
    numpy RNG state.
  - ExtractionResult invariants (len(vector) == len(feature_names), finite).
"""
from __future__ import annotations

import numpy as np
import pytest

from Qmes import get_extractor, ClassificationExtractor, RegressionExtractor
from Qmes.extractors.base import ExtractionResult


# ── factory ───────────────────────────────────────────────────────────────
def test_get_extractor_returns_correct_type():
    assert isinstance(get_extractor("classification"), ClassificationExtractor)
    assert isinstance(get_extractor("regression"), RegressionExtractor)


@pytest.mark.parametrize("alias", ["classification", "Classification", "CLASSIFICATION"])
def test_get_extractor_normalizes_task_name(alias):
    assert get_extractor(alias).task_type == "classification"


def test_get_extractor_unknown_raises():
    with pytest.raises(ValueError, match="Unknown task type"):
        get_extractor("timeseries")


# ── dimension ─────────────────────────────────────────────────────────────
def test_clf_dimension_is_22(tiny_clf):
    X, y = tiny_clf
    res = get_extractor("classification").extract(X, y)
    assert res.dim == 22
    assert len(res.feature_names) == 22
    assert res.task_type == "classification"


def test_reg_dimension_is_12(tiny_reg):
    X, y = tiny_reg
    res = get_extractor("regression").extract(X, y)
    assert res.dim == 12
    assert len(res.feature_names) == 12
    assert res.task_type == "regression"


# ── determinism ───────────────────────────────────────────────────────────
# NOTE: determinism is guaranteed to floating-point tolerance, not bit-for-
# bit. The stochastic metrics (l3/n4/s4) are seed-averaged, but the `hubs`
# feature is an iterative igraph hub-score solve that can wobble by ~1 ULP on
# near-degenerate graphs. Tolerance-based asserts keep these tests from being
# flaky while still catching any *real* (non-numerical) non-determinism.
_DET_ATOL = 1e-9


def test_clf_extract_is_deterministic(tiny_clf):
    X, y = tiny_clf
    ext = get_extractor("classification")
    v1 = ext.extract(X, y).vector
    v2 = ext.extract(X, y).vector
    np.testing.assert_allclose(v1, v2, atol=_DET_ATOL)


def test_reg_extract_is_deterministic(tiny_reg):
    X, y = tiny_reg
    ext = get_extractor("regression")
    v1 = ext.extract(X, y).vector
    v2 = ext.extract(X, y).vector
    np.testing.assert_allclose(v1, v2, atol=_DET_ATOL)


def test_extract_independent_of_global_rng(tiny_clf):
    """The internal seed-averaging must make output independent of the
    caller's global numpy RNG state — otherwise determinism is illusory."""
    X, y = tiny_clf
    ext = get_extractor("classification")

    np.random.seed(999)
    v_a = ext.extract(X, y).vector
    np.random.seed(0)
    v_b = ext.extract(X, y).vector
    np.testing.assert_allclose(v_a, v_b, atol=_DET_ATOL)


# ── invariants ────────────────────────────────────────────────────────────
def test_vector_is_finite(tiny_clf, tiny_reg):
    for task, (X, y) in [("classification", tiny_clf), ("regression", tiny_reg)]:
        v = get_extractor(task).extract(X, y).vector
        assert np.all(np.isfinite(v)), f"{task} produced non-finite meta-features"


def test_extraction_result_length_mismatch_rejected():
    with pytest.raises(ValueError, match="length"):
        ExtractionResult(
            vector=np.zeros(3),
            feature_names=["a", "b"],   # len 2 != 3
            task_type="classification",
        )


def test_missing_y_raises(tiny_clf):
    X, _ = tiny_clf
    with pytest.raises(ValueError, match="requires y"):
        get_extractor("classification").extract(X, None)


def test_to_dict_roundtrip(tiny_reg):
    X, y = tiny_reg
    res = get_extractor("regression").extract(X, y)
    d = res.to_dict()
    assert set(d.keys()) == set(res.feature_names)
    assert len(d) == res.dim