"""Qmes/data/clf/inference.py

Hold-out dataset pool for classification inference (no overlap with training datasets).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from Qmes.config import MAX_SAMPLES
from Qmes.data.preprocessing import encode_categoricals, impute_and_cast

logger = logging.getLogger(__name__)

N_SAMPLES = MAX_SAMPLES
RANDOM_STATE = 42

# ── UCI: (id, signal) ────────────────────────────────────────────
_UCI = {
    "Inf_Haberman":      (43,  True),
    "Inf_BloodTransf":   (176, True),
    "Inf_GermanCredit":  (144, True),
    "Inf_BreastCoimbra": (451, True),
    "Inf_Raisin":        (850, True),
}

# ── OpenML: (id, signal) ─────────────────────────────────────────
_OPENML = {
    "Inf_Banana":  (1460, True),
    "Inf_Phoneme": (1489, True),
}


def _subsample(X: np.ndarray, y: np.ndarray, n=N_SAMPLES, rs=RANDOM_STATE):
    if len(X) <= n:
        return X, y
    from sklearn.model_selection import train_test_split
    _, X_s, _, y_s = train_test_split(
        X, y, test_size=n, stratify=y, random_state=rs
    )
    return X_s, y_s


def _binarize_y(y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"Expected binary, got {len(classes)} classes: {classes}")
    return (y == classes[1]).astype(int)


# ── Synthetic  ──────────────────
def _make_xor(n_samples=300, noise=0.15, n_noise_dims=6, random_state=99):
    rng = np.random.default_rng(random_state)
    n = n_samples // 4
    centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 1, 1, 0])
    X_list, y_list = [], []
    for c, lab in zip(centers, labels):
        X_list.append(rng.normal(loc=c, scale=noise, size=(n, 2)))
        y_list.append(np.full(n, lab))
    X, y = np.vstack(X_list), np.concatenate(y_list)
    if n_noise_dims > 0:
        X = np.hstack([X, rng.normal(0, 0.5, size=(len(y), n_noise_dims))])
    return X.astype(np.float64), y.astype(int)


def _make_two_spirals(n_samples=300, noise=0.4, random_state=99):
    rng = np.random.default_rng(random_state)
    n = n_samples // 2
    theta = np.sqrt(rng.uniform(0, 1, n)) * 3 * np.pi
    r = theta + np.pi
    x_a, y_a = r*np.cos(theta)+rng.normal(0, noise, n), r*np.sin(theta)+rng.normal(0, noise, n)
    x_b, y_b = -r*np.cos(theta)+rng.normal(0, noise, n), -r*np.sin(theta)+rng.normal(0, noise, n)
    X = np.vstack([np.c_[x_a, y_a], np.c_[x_b, y_b]])
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    return X.astype(np.float64), y

_SYNTHETIC = {
    "Inf_XOR_8D":        (lambda: _make_xor(300, 0.15, 6),     False),
    "Inf_Spiral_xlarge": (lambda: _make_two_spirals(300, 0.4), True),
}

def load_inference_classification(
    verbose: bool = True,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, bool]]:

    from ucimlrepo import fetch_ucirepo
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder

    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    signal_flags: dict[str, bool] = {}

    def _log(name, X, y):
        ratio = np.bincount(y) / len(y)
        logger.info("  %-20s %s  class=%s", name, X.shape,
                    np.round(ratio, 2).tolist())

    # ── UCI ──────────────────────────────────────────────────────
    for name, (uci_id, sig) in _UCI.items():
        try:
            d = fetch_ucirepo(id=uci_id)
            X = d.data.features.copy()
            y = d.data.targets.iloc[:, 0]
            X = encode_categoricals(X)                        
            X = impute_and_cast(X).values.astype(np.float64)  
            y = _binarize_y(y.values)
            X, y = _subsample(X, y)
            datasets[name] = (X, y)
            signal_flags[name] = sig
            if verbose:
                _log(name, X, y)
        except Exception:
            logger.exception("FAILED loading %s (UCI id=%d)", name, uci_id)

    # ── OpenML ───────────────────────────────────────────────────
    for name, (oml_id, sig) in _OPENML.items():
        try:
            d = fetch_openml(data_id=oml_id, as_frame=True)
            X = d.data.copy()
            X = encode_categoricals(X)                         
            X = impute_and_cast(X).values.astype(np.float64)  
            y = _binarize_y(np.asarray(d.target))
            X, y = _subsample(X, y)
            datasets[name] = (X, y)
            signal_flags[name] = sig
            if verbose:
                _log(name, X, y)
        except Exception:
            logger.exception("FAILED loading %s (OpenML id=%d)", name, oml_id)

    # ── Synthetic ────────────────────────────────────────────────
    for name, (gen, sig) in _SYNTHETIC.items():
        X, y = gen()
        datasets[name] = (X, y)
        signal_flags[name] = sig
        if verbose:
            _log(name, X, y)

    if verbose:
        n_sig = sum(signal_flags.values())
        logger.info("Loaded %d datasets (%d signal, %d ceiling)",
                    len(datasets), n_sig, len(datasets) - n_sig)
    return datasets, signal_flags