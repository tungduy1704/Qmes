"""Tests for the quantum-Oracle evaluators.

These run the real Qsun quantum kernel, but on tiny datasets it is fast
(~0.3 s for all 7 circuits at n=60), so no slow marker is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from Qmes import get_evaluator
from Qmes.evaluators import filter_degenerate_datasets


# ── metadata ──────────────────────────────────────────────────────────────
def test_clf_evaluator_metadata():
    ev = get_evaluator("classification")
    assert ev.task_type == "classification"
    assert ev.metric_name == "MCC"


def test_reg_evaluator_metadata():
    ev = get_evaluator("regression")
    assert ev.task_type == "regression"
    assert ev.metric_name == "R2"


def test_get_evaluator_unknown_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown task type"):
        get_evaluator("clustering")


# ── single-circuit evaluation ─────────────────────────────────────────────
def test_clf_evaluate_circuit_keys_and_finite(tiny_clf):
    X, y = tiny_clf
    res = get_evaluator("classification", n_splits=3).evaluate_circuit(X, y, "RY")
    for key in ("mean_mcc", "std_mcc", "mean_acc", "std_acc", "mean_f1", "std_f1"):
        assert key in res
        assert np.isfinite(res[key])
    assert -1.0 <= res["mean_mcc"] <= 1.0


def test_reg_evaluate_circuit_keys_and_finite(tiny_reg):
    X, y = tiny_reg
    res = get_evaluator("regression", n_splits=3).evaluate_circuit(X, y, "RY")
    for key in ("mean_r2", "std_r2"):
        assert key in res
        assert np.isfinite(res[key])


# ── all-circuit evaluation ────────────────────────────────────────────────
def test_evaluate_all_covers_every_circuit(tiny_clf, circuit_names):
    X, y = tiny_clf
    scores = get_evaluator("classification", n_splits=3).evaluate_all(X, y)
    assert set(scores.keys()) == set(circuit_names)
    # every circuit produced a usable MCC
    for c in circuit_names:
        assert "mean_mcc" in scores[c]


def test_evaluate_circuit_is_deterministic(tiny_clf):
    """Fixed random_state -> identical CV folds -> identical scores."""
    X, y = tiny_clf
    ev = get_evaluator("classification", n_splits=3, random_state=42)
    a = ev.evaluate_circuit(X, y, "RY")["mean_mcc"]
    b = ev.evaluate_circuit(X, y, "RY")["mean_mcc"]
    assert a == b


# ── pivot building + degenerate filter ────────────────────────────────────
def test_build_pivot_shape(tiny_clf, circuit_names):
    X, y = tiny_clf
    datasets = {"ds0": (X, y)}
    pivot = get_evaluator("classification", n_splits=3).build_pivot(datasets)
    assert list(pivot.columns) == ["ds0"]
    assert set(pivot.index) == set(circuit_names)


def test_filter_degenerate_datasets():
    # ds_good has spread; ds_noise is all-low; ds_ceiling is all-high.
    pivot = pd.DataFrame(
        {
            "ds_good":    [0.1, 0.5, 0.9],
            "ds_noise":   [0.01, 0.02, 0.05],
            "ds_ceiling": [0.995, 0.998, 1.0],
        },
        index=["c0", "c1", "c2"],
    )
    clean, removed = filter_degenerate_datasets(pivot)
    assert "ds_good" in clean.columns
    assert "ds_noise" in removed
    assert "ds_ceiling" in removed