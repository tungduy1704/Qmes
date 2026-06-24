"""Qmes/data/reg/train.py

Data loader for tabular regression datasets.
Returns RAW (X, y) — scaling/normalization is NOT performed here.
Extractor and Evaluator handle their own scaling as needed.

Subsampling policy: each dataset is capped at MAX_SAMPLES=600 (random,
unstratified — the target is continuous) at load time. Both Extractor and
Evaluator receive pre-capped data — no additional subsampling is done
downstream.

Sources:
    - sklearn built-ins
    - UCI ML Repository (via ucimlrepo)
    - CSV files
    - Synthetic (sklearn.datasets generators)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import resample

from Qmes.config import MAX_SAMPLES
from Qmes.data.preprocessing import encode_categoricals, impute_and_cast

logger = logging.getLogger(__name__)

DATA_DIR = Path(
    os.environ.get(
        "QMES_DATA_DIR",
        Path(__file__).resolve().parents[3] / "datasets",
    )
)

CACHE_DIR = Path(
    os.environ.get(
        "QMES_CACHE_DIR_REG",
        Path(__file__).resolve().parents[3] / "data" / "cache" / "regression",
    )
)

def save_cache(datasets: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index = {}
    for name, (X, y) in datasets.items():
        safe_name = name.replace(" ", "_").replace("/", "-")
        np.savez(CACHE_DIR / f"{safe_name}.npz", X=X, y=y)
        index[safe_name] = name

    import json
    with open(CACHE_DIR / "index.json", "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d datasets to cache: %s", len(datasets), CACHE_DIR)


def load_cache() -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    import json
    index_path = CACHE_DIR / "index.json"
    if not CACHE_DIR.exists() or not index_path.exists():
        return None
    with open(index_path) as f:
        index = json.load(f)
    datasets = {}
    for f in sorted(CACHE_DIR.glob("*.npz")):
        safe_name = f.stem
        name = index.get(safe_name, safe_name)
        data = np.load(f)
        datasets[name] = (data["X"], data["y"])
    logger.info("Loaded %d datasets from cache: %s", len(datasets), CACHE_DIR)
    return datasets

# ── Dataset registries ───────────────────────────────────────────────────────
_SKLEARN_DATASETS: dict[str, dict] = {
    "Diabetes":           {"loader": "load_diabetes"},
    "California Housing": {"fetcher": "fetch_california_housing"},
}

# UCI datasets 
_UCI_DATASETS: dict[str, dict] = {
    "Abalone":                  {"data_id": 1},
    "Auto MPG":                 {"data_id": 9},
    "Automobile":               {"data_id": 10,  "target_col": "price"},
    "Concrete":                 {"data_id": 165},
    "Energy Efficiency":        {"data_id": 242, "target_col": "Y1"},
    "Wine Quality Red":         {"data_id": 186},
    "Airfoil":                  {"data_id": 291},
    "Real Estate":              {"data_id": 477},
    "Bike Sharing":             {"data_id": 275, "target_col": "cnt"},
    "Student Math":             {"data_id": 320, "target_col": "G3"},
    "Servo":                    {"data_id": 87},
    "Forest Fires":             {"data_id": 162, "target_col": "area"},
    "Communities Crime":        {"data_id": 183},
    "Computer Hardware":        {"data_id": 29,  "target_col": "PRP"},
    "Parkinson Telemonitoring": {"data_id": 189, "target_col": "motor_UPDRS"},
    "Power Plant":              {"data_id": 294},
    "Facebook Metrics":         {"data_id": 368, "target_col": "like"},
    "Appliances Energy":        {"data_id": 374},
    "Superconductivity":        {"data_id": 464},
    "Liver Disorders":          {"data_id": 60},
    "Thermography":             {"data_id": 925},
    "Solar Flare":              {"data_id": 89},
    "Bias Correction":          {"data_id": 763, "target_col": "M"},
    "Beijing PM2.5":            {"data_id": 381, "target_col": "pm2.5"},
    "Metro Interstate":         {"data_id": 492, "target_col": "traffic_volume"},
    "Electrical Grid":          {"data_id": 471, "target_col": "stab"},
}

_CSV_DATASETS: dict[str, dict] = {}

_SYNTHETIC_DATASETS: dict[str, dict] = {
    # ── Baseline linear ───────────────────────────────────────────────────────
    "Syn_Reg_2f_linear":        {"type": "make_regression",
                                 "params": {"n_samples": 300, "n_features": 2,
                                            "n_informative": 2, "noise": 0.1}},
    "Syn_Reg_5f_linear":        {"type": "make_regression",
                                 "params": {"n_samples": 300, "n_features": 5,
                                            "n_informative": 5, "noise": 0.1}},
    "Syn_Reg_10f_linear":       {"type": "make_regression",
                                 "params": {"n_samples": 400, "n_features": 10,
                                            "n_informative": 10, "noise": 0.1}},

    # ── Redundant features ────────────────────────────────────────────────────
    "Syn_Reg_2inf_0red":        {"type": "make_regression",
                                 "params": {"n_samples": 300, "n_features": 4,
                                            "n_informative": 2, "noise": 0.5}},
    "Syn_Reg_2inf_4red":        {"type": "make_regression",
                                 "params": {"n_samples": 300, "n_features": 8,
                                            "n_informative": 2, "noise": 0.5}},
    "Syn_Reg_2inf_8red":        {"type": "make_regression",
                                 "params": {"n_samples": 300, "n_features": 12,
                                            "n_informative": 2, "noise": 0.5}},

    # ── Sample size ───────────────────────────────────────────────────────────
    "Syn_Reg_small_n50":        {"type": "make_regression",
                                 "params": {"n_samples": 50,  "n_features": 4,
                                            "n_informative": 3, "noise": 0.5}},
    "Syn_Reg_medium_n200":      {"type": "make_regression",
                                 "params": {"n_samples": 200, "n_features": 4,
                                            "n_informative": 3, "noise": 0.5}},
    "Syn_Reg_tiny_n30":         {"type": "make_regression",
                                 "params": {"n_samples": 30,  "n_features": 3,
                                            "n_informative": 2, "noise": 0.5}},

    # ── High dimensional ──────────────────────────────────────────────────────
    "Syn_Reg_highdim_10f":      {"type": "make_regression",
                                 "params": {"n_samples": 400, "n_features": 10,
                                            "n_informative": 4, "noise": 1.0}},
    "Syn_Reg_highdim_20f":      {"type": "make_regression",
                                 "params": {"n_samples": 400, "n_features": 20,
                                            "n_informative": 5, "noise": 1.0}},
    "Syn_Reg_highdim_30f":      {"type": "make_regression",
                                 "params": {"n_samples": 400, "n_features": 30,
                                            "n_informative": 5, "noise": 1.0}},
    "Syn_Reg_highdim_40f":      {"type": "make_regression",
                                 "params": {"n_samples": 400, "n_features": 40,
                                            "n_informative": 6, "noise": 1.0}},

    # ── Sparse informative features ───────────────────────────────────────────
    "Syn_Reg_sparse_1inf":      {"type": "make_regression",
                                 "params": {"n_samples": 300, "n_features": 10,
                                            "n_informative": 1, "noise": 0.5}},
    "Syn_Reg_sparse_2inf":      {"type": "make_regression",
                                 "params": {"n_samples": 300, "n_features": 10,
                                            "n_informative": 2, "noise": 0.5}},
    "Syn_Reg_sparse_5inf":      {"type": "make_regression",
                                 "params": {"n_samples": 300, "n_features": 10,
                                            "n_informative": 5, "noise": 0.5}},
    "Syn_Reg_sparse_10inf":     {"type": "make_regression",
                                 "params": {"n_samples": 300, "n_features": 20,
                                            "n_informative": 10, "noise": 0.5}},

    # ── High redundancy + noise ───────────────────────────────────────────────
    "Syn_Reg_2inf_10red_noise": {"type": "make_regression",
                                 "params": {"n_samples": 300, "n_features": 15,
                                            "n_informative": 2, "noise": 5.0}},
    "Syn_Reg_3inf_12red":       {"type": "make_regression",
                                 "params": {"n_samples": 300, "n_features": 15,
                                            "n_informative": 3, "noise": 1.0}},

    # ── Multicollinear ────────────────────────────────────────────────────────
    "Syn_Reg_multicol_low":     {"type": "make_regression",
                                 "params": {"n_samples": 300, "n_features": 6,
                                            "n_informative": 4, "effective_rank": 2,
                                            "noise": 0.3}},
    "Syn_Reg_multicol_high":    {"type": "make_regression",
                                 "params": {"n_samples": 300, "n_features": 10,
                                            "n_informative": 3, "effective_rank": 2,
                                            "noise": 0.3}},

    # ── Single informative feature ────────────────────────────────────────────
    "Syn_Reg_1inf_3f":          {"type": "make_regression",
                                 "params": {"n_samples": 300, "n_features": 3,
                                            "n_informative": 1, "noise": 0.3}},
    "Syn_Reg_1inf_8f":          {"type": "make_regression",
                                 "params": {"n_samples": 300, "n_features": 8,
                                            "n_informative": 1, "noise": 0.3}},

    # ── Nonlinear: quadratic ──────────────────────────────────────────────────
    "Syn_Reg_quad_2f":          {"type": "quadratic",
                                 "params": {"n_samples": 300, "n_features": 2,
                                            "noise": 0.1}},
    "Syn_Reg_quad_4f":          {"type": "quadratic",
                                 "params": {"n_samples": 300, "n_features": 4,
                                            "noise": 0.1}},
    "Syn_Reg_quad_6f":          {"type": "quadratic",
                                 "params": {"n_samples": 300, "n_features": 6,
                                            "noise": 0.05}},
    "Syn_Reg_quad_8f":          {"type": "quadratic",
                                 "params": {"n_samples": 400, "n_features": 8,
                                            "noise": 0.5}},
    "Syn_Reg_quad_highdim":     {"type": "quadratic",
                                 "params": {"n_samples": 400, "n_features": 12,
                                            "noise": 0.3}},
    "Syn_Reg_quad_noise":       {"type": "quadratic",
                                 "params": {"n_samples": 300, "n_features": 3,
                                            "noise": 1.0}},
    "Syn_Reg_quad_lownoise":    {"type": "quadratic",
                                 "params": {"n_samples": 300, "n_features": 3,
                                            "noise": 0.01}},

    # ── Nonlinear: sinusoidal ─────────────────────────────────────────────────
    "Syn_Reg_sin_1f":           {"type": "sinusoidal",
                                 "params": {"n_samples": 300, "n_features": 3,
                                            "n_active": 1, "noise": 0.1}},
    "Syn_Reg_sin_2f":           {"type": "sinusoidal",
                                 "params": {"n_samples": 300, "n_features": 4,
                                            "n_active": 2, "noise": 0.1}},
    "Syn_Reg_sin_3f_clean":     {"type": "sinusoidal",
                                 "params": {"n_samples": 300, "n_features": 3,
                                            "n_active": 3, "noise": 0.0}},
    "Syn_Reg_sin_highfreq":     {"type": "sinusoidal",
                                 "params": {"n_samples": 300, "n_features": 4,
                                            "n_active": 4, "noise": 0.05}},
    "Syn_Reg_sin_noise":        {"type": "sinusoidal",
                                 "params": {"n_samples": 300, "n_features": 4,
                                            "n_active": 2, "noise": 1.0}},
    "Syn_Reg_sin_highdim":      {"type": "sinusoidal",
                                 "params": {"n_samples": 400, "n_features": 15,
                                            "n_active": 4, "noise": 0.5}},

    # ── Nonlinear: interaction ────────────────────────────────────────────────
    "Syn_Reg_interact_2f":      {"type": "interaction",
                                 "params": {"n_samples": 300, "n_features": 4,
                                            "n_interact": 2, "noise": 0.2}},
    "Syn_Reg_interact_3f":      {"type": "interaction",
                                 "params": {"n_samples": 300, "n_features": 6,
                                            "n_interact": 3, "noise": 0.2}},
    "Syn_Reg_interact_4f_lownoise": {"type": "interaction",
                                 "params": {"n_samples": 300, "n_features": 6,
                                            "n_interact": 4, "noise": 0.05}},
    "Syn_Reg_interact_5f":      {"type": "interaction",
                                 "params": {"n_samples": 300, "n_features": 8,
                                            "n_interact": 5, "noise": 0.1}},
    "Syn_Reg_interact_clean":   {"type": "interaction",
                                 "params": {"n_samples": 300, "n_features": 6,
                                            "n_interact": 6, "noise": 0.0}},
    "Syn_Reg_interact_noise":   {"type": "interaction",
                                 "params": {"n_samples": 300, "n_features": 5,
                                            "n_interact": 2, "noise": 1.5}},
    "Syn_Reg_interact_highdim": {"type": "interaction",
                                 "params": {"n_samples": 400, "n_features": 10,
                                            "n_interact": 4, "noise": 0.5}},

    # ── Nonlinear: Friedman ───────────────────────────────────────────────────
    "Syn_Reg_friedman1":        {"type": "friedman1",
                                 "params": {"n_samples": 300, "n_features": 10,
                                            "noise": 0.0}},
    "Syn_Reg_friedman1_noise":  {"type": "friedman1",
                                 "params": {"n_samples": 300, "n_features": 10,
                                            "noise": 1.0}},
    "Syn_Reg_friedman1_5f":     {"type": "friedman1",
                                 "params": {"n_samples": 300, "n_features": 5,
                                            "noise": 0.5}},
    "Syn_Reg_friedman1_15f":    {"type": "friedman1",
                                 "params": {"n_samples": 300, "n_features": 15,
                                            "noise": 0.5}},
    "Syn_Reg_friedman1_20f":    {"type": "friedman1",
                                 "params": {"n_samples": 300, "n_features": 20,
                                            "noise": 0.5}},
    "Syn_Reg_friedman2":        {"type": "friedman2",
                                 "params": {"n_samples": 300, "noise": 0.0}},
    "Syn_Reg_friedman2_noise":  {"type": "friedman2",
                                 "params": {"n_samples": 300, "noise": 1.0}},
    "Syn_Reg_friedman3":        {"type": "friedman3",
                                 "params": {"n_samples": 300, "noise": 0.0}},
    "Syn_Reg_friedman3_noise":  {"type": "friedman3",
                                 "params": {"n_samples": 300, "noise": 0.3}},

    # ── Heteroscedastic noise ─────────────────────────────────────────────────
    "Syn_Reg_hetero_low":       {"type": "heteroscedastic",
                                 "params": {"n_samples": 300, "n_features": 4,
                                            "noise_scale": 0.5}},
    "Syn_Reg_hetero_high":      {"type": "heteroscedastic",
                                 "params": {"n_samples": 300, "n_features": 4,
                                            "noise_scale": 2.0}},
    "Syn_Reg_hetero_highdim":   {"type": "heteroscedastic",
                                 "params": {"n_samples": 300, "n_features": 8,
                                            "noise_scale": 1.0}},
    "Syn_Reg_hetero_10f":       {"type": "heteroscedastic",
                                 "params": {"n_samples": 300, "n_features": 10,
                                            "noise_scale": 1.5}},
    "Syn_Reg_hetero_corr":      {"type": "heteroscedastic",
                                 "params": {"n_samples": 300, "n_features": 6,
                                            "noise_scale": 1.5}},
    "Syn_Reg_hetero_small_n":   {"type": "heteroscedastic",
                                 "params": {"n_samples": 100, "n_features": 4,
                                            "noise_scale": 1.0}},

    # ── Outlier-heavy target ──────────────────────────────────────────────────
    "Syn_Reg_outlier_low":      {"type": "outlier",
                                 "params": {"n_samples": 300, "n_features": 4,
                                            "outlier_frac": 0.05, "noise": 0.3}},
    "Syn_Reg_outlier_high":     {"type": "outlier",
                                 "params": {"n_samples": 300, "n_features": 4,
                                            "outlier_frac": 0.15, "noise": 0.3}},

    # ── Correlated input features ─────────────────────────────────────────────
    "Syn_Reg_corr_low":         {"type": "correlated",
                                 "params": {"n_samples": 300, "n_features": 5,
                                            "corr": 0.2, "noise": 0.3}},
    "Syn_Reg_corr_high":        {"type": "correlated",
                                 "params": {"n_samples": 300, "n_features": 5,
                                            "corr": 0.9, "noise": 0.3}},
    "Syn_Reg_corr_interact":    {"type": "correlated",
                                 "params": {"n_samples": 300, "n_features": 6,
                                            "corr": 0.7, "noise": 0.5}},
    "Syn_Reg_corr_highdim":     {"type": "correlated",
                                 "params": {"n_samples": 300, "n_features": 10,
                                            "corr": 0.6, "noise": 0.5}},
    "Syn_Reg_corr_quad":        {"type": "correlated",
                                 "params": {"n_samples": 300, "n_features": 6,
                                            "corr": 0.8, "noise": 0.1}},
    "Syn_Reg_corr_sin":         {"type": "correlated",
                                 "params": {"n_samples": 300, "n_features": 5,
                                            "corr": 0.5, "noise": 0.2}},

    # ── Mixed ─────────────────────────────────────────────────────────────────
    "Syn_Reg_mixed_quad_10f":   {"type": "quadratic",
                                 "params": {"n_samples": 400, "n_features": 10,
                                            "noise": 0.5}},
    "Syn_Reg_mixed_sin_10f":    {"type": "sinusoidal",
                                 "params": {"n_samples": 400, "n_features": 10,
                                            "n_active": 3, "noise": 0.5}},
    "Syn_Reg_mixed_interact_sin": {"type": "interaction",
                                 "params": {"n_samples": 300, "n_features": 6,
                                            "n_interact": 3, "noise": 0.3}},
}

# ── Custom synthetic generators ──────────────────────────────────────────────
def _gen_quadratic(
    n_samples: int,
    n_features: int,
    noise: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """y = sum(x_i^2) + cross terms x_0*x_1 + noise."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    y = np.sum(X ** 2, axis=1) + X[:, 0] * X[:, 1]
    if noise > 0:
        y += rng.randn(n_samples) * noise * np.std(y)
    return X.astype(np.float64), y.astype(np.float64)


def _gen_sinusoidal(
    n_samples: int,
    n_features: int,
    n_active: int = 2,
    noise: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """y = sum_{i < n_active} sin(2*pi*x_i) + noise."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1, 1, size=(n_samples, n_features))
    n_active = min(n_active, n_features)
    y = np.sum(np.sin(2 * np.pi * X[:, :n_active]), axis=1)
    if noise > 0:
        y += rng.randn(n_samples) * noise * np.std(y)
    return X.astype(np.float64), y.astype(np.float64)

def _gen_interaction(
    n_samples: int,
    n_features: int,
    n_interact: int = 2,
    noise: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """y = linear term + pairwise interactions among first n_interact features + noise."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    coef = rng.randn(n_features)
    y = X @ coef
    n_interact = min(n_interact, n_features)
    for i in range(n_interact):
        for j in range(i + 1, n_interact):
            y += X[:, i] * X[:, j]
    if noise > 0:
        y += rng.randn(n_samples) * noise * np.std(y)
    return X.astype(np.float64), y.astype(np.float64)


def _gen_heteroscedastic(
    n_samples: int,
    n_features: int,
    noise_scale: float = 1.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """y = X @ coef + noise * |x_0| ."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    coef = rng.randn(n_features)
    y = X @ coef + rng.randn(n_samples) * noise_scale * np.abs(X[:, 0])
    return X.astype(np.float64), y.astype(np.float64)


def _gen_outlier(
    n_samples: int,
    n_features: int,
    outlier_frac: float = 0.05,
    noise: float = 0.3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    coef = rng.randn(n_features)
    y = X @ coef + rng.randn(n_samples) * noise
    n_outliers = max(1, int(n_samples * outlier_frac))
    idx = rng.choice(n_samples, n_outliers, replace=False)
    y[idx] += rng.randn(n_outliers) * 10 * np.std(y)
    return X.astype(np.float64), y.astype(np.float64)


def _gen_correlated(
    n_samples: int,
    n_features: int,
    corr: float = 0.5,
    noise: float = 0.3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    cov = np.full((n_features, n_features), corr)
    np.fill_diagonal(cov, 1.0)
    X = rng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    coef = rng.randn(n_features)
    y = X @ coef + rng.randn(n_samples) * noise
    return X.astype(np.float64), y.astype(np.float64)

_CUSTOM_GENERATORS = {
    "quadratic":      _gen_quadratic,
    "sinusoidal":     _gen_sinusoidal,
    "interaction":    _gen_interaction,
    "heteroscedastic": _gen_heteroscedastic,
    "outlier":        _gen_outlier,
    "correlated":     _gen_correlated,
}

# ── Internal helpers ─────────────────────────────────────────────────────────
def _random_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int = MAX_SAMPLES,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= max_samples:
        return X, y
    X_sub, y_sub = resample(
        X, y,
        n_samples=max_samples,
        replace=False,
        random_state=random_state,
    )
    return X_sub, y_sub

def _load_uci(
    data_id: int,
    target_col: str | None = None,
    target_map: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    from ucimlrepo import fetch_ucirepo

    repo = fetch_ucirepo(id=data_id)
    X_df = repo.data.features.copy()
    original_df = repo.data.original

    if target_col is not None:
        targets_df = repo.data.targets
        if targets_df is not None and target_col in targets_df.columns:
            y_series = targets_df[target_col]
            X_df = X_df.drop(columns=[target_col], errors="ignore") 
        elif target_col in original_df.columns:
            y_series = original_df[target_col]
            X_df = X_df.drop(columns=[target_col], errors="ignore")
        else:
            raise ValueError(
                f"UCI {data_id}: target_col '{target_col}' không tìm thấy. "
                f"original columns: {list(original_df.columns)}"
            )
    else:
        targets_df = repo.data.targets
        if targets_df is None:
            raise ValueError(
                f"UCI {data_id}: repo.data.targets is None. "
                "Chỉ định target_col để lấy target từ repo.data.original."
            )
        y_series = targets_df.iloc[:, 0]

    if target_map is not None:
        y_series = y_series.astype(str).str.strip().map(target_map)

    y = pd.to_numeric(y_series, errors="coerce").values.astype(np.float64)

    nan_mask = np.isnan(y)
    if nan_mask.any():
        if nan_mask.all():
            raise ValueError(f"UCI {data_id}: all target values are NaN")
        logger.warning(
            "UCI %d: dropped %d rows with NaN target", data_id, int(nan_mask.sum())
        )
        X_df = X_df[~nan_mask].reset_index(drop=True)
        y = y[~nan_mask]

    X_df = encode_categoricals(X_df)
    X_df = impute_and_cast(X_df, strategy="median")
    return X_df.values.astype(np.float64), y

def _load_csv(
    path: Path,
    header: int | None = 0,
    columns: list[str] | None = None,
    target_col: str | int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=header)
    if columns is not None:
        df.columns = columns
    if isinstance(target_col, int):
        target_col = df.columns[target_col]
    y = pd.to_numeric(df[target_col], errors="coerce").values.astype(np.float64)
    X_df = df.drop(columns=target_col)
    X_df = encode_categoricals(X_df)
    X_df = impute_and_cast(X_df, strategy="median")
    nan_mask = np.isnan(y)
    if nan_mask.any():
        y[nan_mask] = np.nanmedian(y)
    return X_df.values.astype(np.float64), y


def _load_synthetic(
    type: str,
    params: dict,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if type == "make_regression":
        from sklearn.datasets import make_regression
        p = {k: v for k, v in params.items() if k != "n_targets"}
        X, y = make_regression(**p, random_state=random_state)
        if y.ndim > 1:
            y = y[:, 0]
        return X.astype(np.float64), y.astype(np.float64)

    elif type in ("friedman1", "friedman2", "friedman3"):
        from sklearn.datasets import (
            make_friedman1, make_friedman2, make_friedman3,
        )
        _fn = {"friedman1": make_friedman1,
               "friedman2": make_friedman2,
               "friedman3": make_friedman3}[type]
        X, y = _fn(**params, random_state=random_state)
        return X.astype(np.float64), y.astype(np.float64)

    elif type in _CUSTOM_GENERATORS:
        return _CUSTOM_GENERATORS[type](**params, random_state=random_state)

    else:
        raise ValueError(f"Unknown synthetic type: {type!r}")

# ── Public API ───────────────────────────────────────────────────────────────
def load_regression_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load all regression datasets, subsampled to MAX_SAMPLES.

    Returns
    -------
    dict[str, (X, y)]
        X : ndarray shape (n_samples, n_features), float64, unscaled
            n_samples <= MAX_SAMPLES
        y : ndarray shape (n_samples,), float64
    """
    import sklearn.datasets as skd

    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    skipped: list[str] = []

    # Cache check
    cached = load_cache()
    if cached is not None:
        return cached

    # ── 1. sklearn ───────────────────────────────────────────────────────────
    for name, spec in _SKLEARN_DATASETS.items():
        try:
            if "loader" in spec:
                bunch = getattr(skd, spec["loader"])(as_frame=True)
            else:
                bunch = getattr(skd, spec["fetcher"])(as_frame=True)
            X_df = encode_categoricals(bunch.data)
            X_df = impute_and_cast(X_df, strategy="median")
            X = X_df.values.astype(np.float64)
            y = bunch.target.values.astype(np.float64)
            datasets[name] = _random_subsample(X, y)
        except Exception as e:
            logger.warning("Regression [sklearn] %s: %s", name, e)
            skipped.append(name)

    # ── 2. UCI ───────────────────────────────────────────────────────────────
    for name, spec in _UCI_DATASETS.items():
        spec = spec.copy()
        data_id = spec.pop("data_id")
        try:
            X, y = _load_uci(data_id, **spec)
            datasets[name] = _random_subsample(X, y)
        except Exception as e:
            logger.warning("Regression [uci] %s (id=%d): %s", name, data_id, e)
            skipped.append(name)

    # ── 3. CSV ───────────────────────────────────────────────────────────────
    for name, spec in _CSV_DATASETS.items():
        spec = spec.copy()
        path = DATA_DIR / spec.pop("path")
        if not path.exists():
            logger.warning("Regression [csv] %s: file not found: %s", name, path)
            skipped.append(name)
            continue
        try:
            X, y = _load_csv(path, **spec)
            datasets[name] = _random_subsample(X, y)
        except Exception as e:
            logger.warning("Regression [csv] %s: %s", name, e)
            skipped.append(name)

    # ── 4. Synthetic ─────────────────────────────────────────────────────────
    for name, spec in _SYNTHETIC_DATASETS.items():
        try:
            X, y = _load_synthetic(
                type=spec["type"],
                params=spec["params"],
            )
            datasets[name] = _random_subsample(X, y)
        except Exception as e:
            logger.warning("Regression [synthetic] %s: %s", name, e)
            skipped.append(name)

    logger.info(
        "Regression: loaded %d datasets%s",
        len(datasets),
        f", skipped {len(skipped)}: {skipped}" if skipped else "",
    )
    save_cache(datasets)
    return datasets