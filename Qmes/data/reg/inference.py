"""Qmes/data//reg/inference.py

Hold-out dataset pool for regression inference (no overlap with training pool).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.utils import resample

from Qmes.config import MAX_SAMPLES
from Qmes.data.preprocessing import encode_categoricals, impute_and_cast

logger = logging.getLogger(__name__)


N_SAMPLES = MAX_SAMPLES
RANDOM_STATE = 99   
SYNTH_STATE = 99    

# ── OpenML reals: {name: data_id} ────────────────────────────────
_OPENML = {
    "Inf_Reg_Stock":    223,   
    "Inf_Reg_CpuSmall": 227,   
    "Inf_Reg_Housing":  531,   
}

# ── Synthetic generators (seed = SYNTH_STATE) ────────────────────
def _gen_quadratic(n_samples, n_features, noise, random_state=SYNTH_STATE):
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    y = np.sum(X ** 2, axis=1) + X[:, 0] * X[:, 1]
    if noise > 0:
        y += rng.randn(n_samples) * noise * np.std(y)
    return X.astype(np.float64), y.astype(np.float64)

def _gen_sinusoidal(n_samples, n_features, n_active, noise, random_state=SYNTH_STATE):
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1, 1, size=(n_samples, n_features))
    n_active = min(n_active, n_features)
    y = np.sum(np.sin(2 * np.pi * X[:, :n_active]), axis=1)
    if noise > 0:
        y += rng.randn(n_samples) * noise * np.std(y)
    return X.astype(np.float64), y.astype(np.float64)

def _gen_friedman(kind, n_samples, noise, random_state=SYNTH_STATE):
    from sklearn.datasets import make_friedman2, make_friedman3
    fn = {2: make_friedman2, 3: make_friedman3}[kind]
    X, y = fn(n_samples=n_samples, noise=noise, random_state=random_state)
    return X.astype(np.float64), y.astype(np.float64)

# ── Synthetic registry: {name: generator} ──
_SYNTHETIC = {
    "Inf_Reg_quad_3f":   lambda: _gen_quadratic(300, 3, 0.05),
    "Inf_Reg_quad_6f":   lambda: _gen_quadratic(300, 6, 0.10),
    "Inf_Reg_quad_8f":   lambda: _gen_quadratic(400, 8, 0.30),
    "Inf_Reg_sin_2f":    lambda: _gen_sinusoidal(300, 4, 2, 0.05),
    "Inf_Reg_sin_3f":    lambda: _gen_sinusoidal(300, 4, 3, 0.10),
    "Inf_Reg_friedman2": lambda: _gen_friedman(2, 300, 0.0),
    "Inf_Reg_friedman3": lambda: _gen_friedman(3, 300, 0.0),
}

def _subsample(X, y, n=N_SAMPLES, rs=RANDOM_STATE):
    if len(X) <= n:
        return X, y
    return resample(X, y, n_samples=n, replace=False, random_state=rs)

def _load_openml(data_id: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_openml
    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    X_df = encode_categoricals(bunch.data.copy())
    X_df = impute_and_cast(X_df, strategy="median")
    X = X_df.values.astype(np.float64)
    y = pd.to_numeric(bunch.target, errors="coerce").values.astype(np.float64)
    nan_mask = np.isnan(y)
    if nan_mask.any():
        y[nan_mask] = np.nanmedian(y)
    return X, y

def load_inference_reg_datasets(
    verbose: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    skipped: list[str] = []

    def _log(name, X, y):
        logger.info("  %-22s %s  y∈[%.2f, %.2f]", name, X.shape, y.min(), y.max())

    # ── OpenML (reals) ───────────────────────────────────────────
    for name, data_id in _OPENML.items():
        try:
            X, y = _load_openml(data_id)
            datasets[name] = _subsample(X, y)
            if verbose:
                _log(name, *datasets[name])
        except Exception:
            logger.exception("FAILED loading %s (OpenML id=%d)", name, data_id)
            skipped.append(name)

    # ── Synthetic ────────────────────────────────────────────────
    for name, gen in _SYNTHETIC.items():
        X, y = _subsample(*gen())
        datasets[name] = (X, y)
        if verbose:
            _log(name, X, y)

    if verbose:
        logger.info(
            "Loaded %d inference (regression) datasets%s",
            len(datasets),
            f", skipped {len(skipped)}: {skipped}" if skipped else "",
        )
    return datasets