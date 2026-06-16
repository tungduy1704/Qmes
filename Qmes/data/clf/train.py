"""Qmes/data/clf/train.py

Data loader for tabular binary classification datasets.
Returns RAW (X, y) — scaling/normalization is NOT performed here.
Extractor and Evaluator handle their own scaling as needed.

Subsampling policy: each dataset is capped at MAX_SAMPLES=600 (stratified)
at load time. Both Extractor and Evaluator receive pre-capped data —
no additional subsampling is done downstream.

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
        "QMES_CACHE_DIR",
        Path(__file__).resolve().parents[3] / "data" / "cache" / "classification",
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

MAX_SAMPLES = 600 

# ── Dataset registries ───────────────────────────────────────────────────────
_SKLEARN_DATASETS: dict[str, dict] = {
    "Iris_01":          {"loader": "load_iris",   "classes": [0, 1]},
    "Iris_02":          {"loader": "load_iris",   "classes": [0, 2]},
    "Iris_12":          {"loader": "load_iris",   "classes": [1, 2]},
    "Wine_01":          {"loader": "load_wine",   "classes": [0, 1]},
    "Wine_02":          {"loader": "load_wine",   "classes": [0, 2]},
    "Wine_12":          {"loader": "load_wine",   "classes": [1, 2]},
    "Digits_01":        {"loader": "load_digits", "classes": [0, 1]},
    "Digits_17":        {"loader": "load_digits", "classes": [1, 7]},
    "Digits_35":        {"loader": "load_digits", "classes": [3, 5]},
    "Digits_49":        {"loader": "load_digits", "classes": [4, 9]},
    "Digits_69":        {"loader": "load_digits", "classes": [6, 9]},
    "Breast Cancer":    {"loader": "load_breast_cancer"},
}

# UCI datasets 
_UCI_DATASETS: dict[str, dict] = {
    "Ionosphere":           {"data_id": 52},
    "Statlog Heart":        {"data_id": 145},
    "Mushroom":             {"data_id": 73},
    "Monks":                {"data_id": 70},
    "Credit Approval":      {"data_id": 27},
    "Australian":           {"data_id": 143},
    "Wdbc":                 {"data_id": 17},
    "Spect Heart":          {"data_id": 95},
    "Breast-w":             {"data_id": 15},
    "Chronic Kidney":       {"data_id": 336},
    "Glass":                {"data_id": 42,  "target_map": {"1": 0, "2": 1}},
    "Diabetes Early":       {"data_id": 529},
    "Cirrhosis":            {"data_id": 878, "target_map": {"C": 0, "D": 1}},
    "Kidney Risk":          {"data_id": 857},
    "Sonar":                {"data_id": 151},
    "Thyroid Cancer":       {"data_id": 915},
    "Magic Gamma":          {"data_id": 159},
    "Mammographic Mass":    {"data_id": 161},
    "Diabetic Retinopathy": {"data_id": 329},
    "Parkinsons":           {"data_id": 174},
    "Phishing Websites":    {"data_id": 327},
    "EEG Eye State":        {"data_id": 264},
    "Vertebral Column":     {"data_id": 212,
                             "target_map": {"Normal": 0, "Spondylolisthesis": 1}},
    "Rice":                 {"data_id": 545},
    "Bank Marketing":       {"data_id": 222},
    "Credit Card":          {"data_id": 350},
    "Iranian Churn":        {"data_id": 563},
    "CDC Diabetes":         {"data_id": 890},
    "Online Shoppers":      {"data_id": 468},
    "Heart Disease":        {"data_id": 45,
                             "target_map": {"0": 0, "1": 1}},
    "Student Dropout":      {"data_id": 697,
                             "target_map": {"Dropout": 1, "Graduate": 0}},
    "Adult Income":         {"data_id": 2,
                             "target_map": {"<=50K": 0, ">50K": 1,
                                            "<=50K.": 0, ">50K.": 1}},
    "Heart Attack":         {"data_id": 519},
    "Smoker Status":        {"data_id": 880},
    "ILPD":                 {"data_id": 225},
    "Spambase":             {"data_id": 94},
    "TicTacToe":            {"data_id": 101},
}

_CSV_DATASETS: dict[str, dict] = {
    "BankNote Auth": {
        "path": "BankNote_Authentication.csv",
    },
    "Pima Diabetes": {
        "path": "pima-indians-diabete.csv",
    },
}

_SYNTHETIC_DATASETS: dict[str, dict] = {
    "Syn_Blobs_8f_easy":    {"generator": "blobs",
                             "params": {"n_samples": 200, "n_features": 8,
                                        "centers": 2, "cluster_std": 0.5}},
    "Syn_Blobs_2f_med":     {"generator": "blobs",
                             "params": {"n_samples": 200, "n_features": 2,
                                        "centers": 2, "cluster_std": 1.5}},
    "Syn_Blobs_2f_hard":    {"generator": "blobs",
                             "params": {"n_samples": 200, "n_features": 2,
                                        "centers": 2, "cluster_std": 3.0}},
    "Syn_Imbal_20_80":      {"generator": "classification",
                             "params": {"n_samples": 300, "n_features": 5,
                                        "n_informative": 3, "n_redundant": 1,
                                        "weights": [0.2, 0.8]}},
    "Syn_Moons_lownoise":   {"generator": "moons",
                             "params": {"n_samples": 200, "noise": 0.05}},
    "Syn_Moons_mednoise":   {"generator": "moons",
                             "params": {"n_samples": 200, "noise": 0.20}},
    "Syn_Moons_highnoise":  {"generator": "moons",
                             "params": {"n_samples": 200, "noise": 0.35}},
    "Syn_Circles_loose":    {"generator": "circles",
                             "params": {"n_samples": 200, "noise": 0.15,
                                        "factor": 0.7}},
    "Syn_Clf_2inf_0red":    {"generator": "classification",
                             "params": {"n_samples": 300, "n_features": 4,
                                        "n_informative": 2, "n_redundant": 0,
                                        "n_repeated": 0}},
    "Syn_Clf_2inf_4red":    {"generator": "classification",
                             "params": {"n_samples": 300, "n_features": 8,
                                        "n_informative": 2, "n_redundant": 4,
                                        "n_repeated": 0}},
    "Syn_Clf_2inf_8red":    {"generator": "classification",
                             "params": {"n_samples": 300, "n_features": 12,
                                        "n_informative": 2, "n_redundant": 8,
                                        "n_repeated": 0}},
    "Syn_HighDim_10f":      {"generator": "classification",
                             "params": {"n_samples": 400, "n_features": 10,
                                        "n_informative": 4, "n_redundant": 3,
                                        "flip_y": 0.05}},
    "Syn_HighDim_20f":      {"generator": "classification",
                             "params": {"n_samples": 400, "n_features": 20,
                                        "n_informative": 5, "n_redundant": 8,
                                        "flip_y": 0.05}},
    "Syn_HighDim_30f":      {"generator": "classification",
                             "params": {"n_samples": 400, "n_features": 30,
                                        "n_informative": 5, "n_redundant": 15,
                                        "flip_y": 0.10}},
    "Syn_Medium_n200":      {"generator": "classification",
                             "params": {"n_samples": 200, "n_features": 4,
                                        "n_informative": 2, "n_redundant": 1}},
    "Syn_Large_n600":       {"generator": "classification",
                             "params": {"n_samples": 600, "n_features": 4,
                                        "n_informative": 2, "n_redundant": 1}},
    "Syn_Corr_low":         {"generator": "classification",
                             "params": {"n_samples": 300, "n_features": 6,
                                        "n_informative": 4, "n_redundant": 0,
                                        "n_clusters_per_class": 1}},
    "Syn_Corr_high":        {"generator": "classification",
                             "params": {"n_samples": 300, "n_features": 6,
                                        "n_informative": 2, "n_redundant": 4,
                                        "n_clusters_per_class": 1}},
    "Syn_Multimodal_1c":    {"generator": "classification",
                             "params": {"n_samples": 300, "n_features": 6,
                                        "n_informative": 3, "n_redundant": 0,
                                        "n_clusters_per_class": 1}},
    "Syn_Multimodal_2c":    {"generator": "classification",
                             "params": {"n_samples": 300, "n_features": 6,
                                        "n_informative": 3, "n_redundant": 0,
                                        "n_clusters_per_class": 2}},
    "Syn_Multimodal_3c":    {"generator": "classification",
                             "params": {"n_samples": 300, "n_features": 6,
                                        "n_informative": 3, "n_redundant": 0,
                                        "n_clusters_per_class": 3}},
    "Syn_Noise_0pct":       {"generator": "classification",
                             "params": {"n_samples": 300, "n_features": 6,
                                        "n_informative": 4, "flip_y": 0.00}},
    "Syn_Noise_10pct":      {"generator": "classification",
                             "params": {"n_samples": 300, "n_features": 6,
                                        "n_informative": 4, "flip_y": 0.10}},
    "Syn_Noise_20pct":      {"generator": "classification",
                             "params": {"n_samples": 300, "n_features": 6,
                                        "n_informative": 4, "flip_y": 0.20}},
    "Syn_Blobs_4f_hard":    {"generator": "blobs",
                             "params": {"n_samples": 300, "n_features": 4,
                                        "centers": 2, "cluster_std": 4.0}},
    "Syn_Blobs_6f_med":     {"generator": "blobs",
                             "params": {"n_samples": 300, "n_features": 6,
                                        "centers": 2, "cluster_std": 2.0}},
    "Syn_Imbal_15_85":      {"generator": "classification",
                             "params": {"n_samples": 400, "n_features": 6,
                                        "n_informative": 3, "n_redundant": 2,
                                        "weights": [0.15, 0.85]}},
    "Syn_Imbal_30_70":      {"generator": "classification",
                             "params": {"n_samples": 300, "n_features": 5,
                                        "n_informative": 3, "n_redundant": 1,
                                        "weights": [0.30, 0.70]}},
    "Syn_Moons_large":      {"generator": "moons",
                             "params": {"n_samples": 500, "noise": 0.15}},
    "Syn_Circles_med":      {"generator": "circles",
                             "params": {"n_samples": 300, "noise": 0.10,
                                        "factor": 0.5}},
    "Syn_Circles_noisy":    {"generator": "circles",
                             "params": {"n_samples": 300, "noise": 0.25,
                                        "factor": 0.6}},
    "Syn_Clf_2inf_12red":   {"generator": "classification",
                             "params": {"n_samples": 400, "n_features": 16,
                                        "n_informative": 2, "n_redundant": 12,
                                        "n_repeated": 0}},
    "Syn_Sparse_20f":       {"generator": "classification",
                             "params": {"n_samples": 400, "n_features": 20,
                                        "n_informative": 2, "n_redundant": 0,
                                        "flip_y": 0.05}},
    "Syn_Sparse_30f":       {"generator": "classification",
                             "params": {"n_samples": 400, "n_features": 30,
                                        "n_informative": 2, "n_redundant": 0,
                                        "flip_y": 0.05}},
    "Syn_XLarge_n1000":     {"generator": "classification",
                             "params": {"n_samples": 1000, "n_features": 4,
                                        "n_informative": 2, "n_redundant": 1}},
    "Syn_Multimodal_4c":    {"generator": "classification",
                             "params": {"n_samples": 400, "n_features": 6,
                                        "n_informative": 3, "n_redundant": 0,
                                        "n_clusters_per_class": 4}},
    "Syn_Noise_30pct":      {"generator": "classification",
                             "params": {"n_samples": 300, "n_features": 6,
                                        "n_informative": 4, "flip_y": 0.30}},
# ── XOR variants ──
    "Syn_XOR_2D_low":       {"generator": "xor",
                             "params": {"n_samples": 300, "noise": 0.05}},
    "Syn_XOR_2D_med":       {"generator": "xor",
                             "params": {"n_samples": 300, "noise": 0.15}},
    "Syn_XOR_2D_high":      {"generator": "xor",
                             "params": {"n_samples": 300, "noise": 0.30}},
    "Syn_XOR_4D":           {"generator": "xor",
                             "params": {"n_samples": 300, "noise": 0.15,
                                        "n_noise_dims": 2}},
    "Syn_XOR_6D":           {"generator": "xor",
                             "params": {"n_samples": 300, "noise": 0.15,
                                        "n_noise_dims": 4}},

    # ── Checkerboard variants ──
    "Syn_Checker_2x2":      {"generator": "checkerboard",
                             "params": {"n_samples": 300, "grid": 2, "noise": 0.08}},
    "Syn_Checker_3x3":      {"generator": "checkerboard",
                             "params": {"n_samples": 300, "grid": 3, "noise": 0.10}},
    "Syn_Checker_4x4":      {"generator": "checkerboard",
                             "params": {"n_samples": 400, "grid": 4, "noise": 0.08}},
    "Syn_Checker_noisy":    {"generator": "checkerboard",
                             "params": {"n_samples": 300, "grid": 3, "noise": 0.20}},

    # ── TwoSpirals variants ──
    "Syn_Spiral_tight":     {"generator": "two_spirals",
                             "params": {"n_samples": 300, "noise": 0.3}},
    "Syn_Spiral_med":       {"generator": "two_spirals",
                             "params": {"n_samples": 300, "noise": 0.5}},
    "Syn_Spiral_loose":     {"generator": "two_spirals",
                             "params": {"n_samples": 300, "noise": 0.8}},
    "Syn_Spiral_large":     {"generator": "two_spirals",
                             "params": {"n_samples": 500, "noise": 0.5}},

    # ── NonlinSubspace variants ──
    "Syn_Nonlin_4D":        {"generator": "nonlin_subspace",
                             "params": {"n_samples": 300, "n_noise_dims": 2}},
    "Syn_Nonlin_6D":        {"generator": "nonlin_subspace",
                             "params": {"n_samples": 300, "n_noise_dims": 4}},
    "Syn_Nonlin_8D":        {"generator": "nonlin_subspace",
                             "params": {"n_samples": 300, "n_noise_dims": 6}},

    # ── Moons rotated variants ──
    "Syn_MoonsRot_3D":      {"generator": "moons_rotated",
                             "params": {"n_samples": 300, "noise": 0.15,
                                        "n_dims": 3}},
    "Syn_MoonsRot_5D":      {"generator": "moons_rotated",
                             "params": {"n_samples": 300, "noise": 0.15,
                                        "n_dims": 5}},
    "Syn_MoonsRot_7D":      {"generator": "moons_rotated",
                             "params": {"n_samples": 300, "noise": 0.15,
                                        "n_dims": 7}},
    "Syn_MoonsRot_noisy":   {"generator": "moons_rotated",
                             "params": {"n_samples": 300, "noise": 0.30,
                                        "n_dims": 5}},
}

# ── Custom synthetic generators ──────────────────────────────────────────────
def _make_xor(n_samples=300, noise=0.15, n_noise_dims=0, random_state=42):
    rng = np.random.default_rng(random_state)
    n = n_samples // 4
    centers = np.array([[0,0],[0,1],[1,0],[1,1]])
    labels  = np.array([0, 1, 1, 0])
    X_list, y_list = [], []
    for c, lab in zip(centers, labels):
        X_list.append(rng.normal(loc=c, scale=noise, size=(n, 2)))
        y_list.append(np.full(n, lab))
    X, y = np.vstack(X_list), np.concatenate(y_list)
    if n_noise_dims > 0:
        X = np.hstack([X, rng.normal(0, 0.5, size=(len(y), n_noise_dims))])
    return X.astype(np.float64), y.astype(int)

def _make_checkerboard(n_samples=300, grid=3, noise=0.1, random_state=42):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(0, grid, size=(n_samples, 2))
    y = ((np.floor(X[:,0]) + np.floor(X[:,1])) % 2).astype(int)
    X += rng.normal(0, noise, size=X.shape)
    return X.astype(np.float64), y

def _make_two_spirals(n_samples=300, noise=0.5, random_state=42):
    rng = np.random.default_rng(random_state)
    n = n_samples // 2
    theta = np.sqrt(rng.uniform(0, 1, n)) * 3 * np.pi
    r = theta + np.pi
    x_a =  r * np.cos(theta) + rng.normal(0, noise, n)
    y_a =  r * np.sin(theta) + rng.normal(0, noise, n)
    x_b = -r * np.cos(theta) + rng.normal(0, noise, n)
    y_b = -r * np.sin(theta) + rng.normal(0, noise, n)
    X = np.vstack([np.c_[x_a, y_a], np.c_[x_b, y_b]])
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    return X.astype(np.float64), y

def _make_nonlin_subspace(n_samples=300, n_noise_dims=4, random_state=42):
    rng = np.random.default_rng(random_state)
    x1 = rng.uniform(-1, 1, n_samples)
    x2 = rng.uniform(-1.5, 1.5, n_samples)
    y = (x2 > np.sin(3 * x1)).astype(int)
    noise_dims = rng.normal(0, 1.0, size=(n_samples, n_noise_dims))
    X = np.column_stack([x1, x2, noise_dims])
    return X.astype(np.float64), y

def _make_moons_rotated(n_samples=300, noise=0.15, n_dims=5, random_state=42):
    from sklearn.datasets import make_moons
    rng = np.random.default_rng(random_state)
    X_2d, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    A = rng.standard_normal((2, n_dims))
    Q, _ = np.linalg.qr(A.T)
    X_nd = X_2d @ Q.T[:2, :]
    X_nd += rng.normal(0, 0.05, size=X_nd.shape)
    return X_nd.astype(np.float64), y

# ── Internal helpers ─────────────────────────────────────────────────────────
def _stratified_subsample(
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
        stratify=y,
        random_state=random_state,
    )
    return X_sub, y_sub

def _load_csv(
    path: Path,
    header: int | None = 0,
    columns: list[str] | None = None,
    target_col: str | int = -1,
    target_map: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=header)
    if columns is not None:
        df.columns = columns
    if isinstance(target_col, int):
        target_col = df.columns[target_col]
    y = df[target_col]
    if target_map is not None:
        y = y.map(target_map)
    y = pd.to_numeric(y, errors="coerce").values.astype(int)
    X_df = df.drop(columns=target_col)
    X_df = encode_categoricals(X_df)
    X_df = impute_and_cast(X_df, strategy="median")
    return X_df.values.astype(np.float64), y

def _load_uci(
    data_id: int,
    target_map: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    from ucimlrepo import fetch_ucirepo

    repo = fetch_ucirepo(id=data_id)
    X_df = repo.data.features.copy()
    y_series = repo.data.targets.iloc[:, 0].astype(str).str.strip()

    if target_map is not None:
        y = y_series.map(target_map)
    else:
        unique_vals = sorted(y_series.unique())
        if len(unique_vals) != 2:
            raise ValueError(
                f"UCI {data_id}: expected 2 classes, "
                f"got {len(unique_vals)}: {unique_vals}"
            )
        y = y_series.map({unique_vals[0]: 0, unique_vals[1]: 1})

    valid_mask = y.notna()
    y = y[valid_mask].reset_index(drop=True)
    X_df = X_df[valid_mask].reset_index(drop=True)

    y = y.values.astype(int)
    X_df = encode_categoricals(X_df)
    X_df = impute_and_cast(X_df, strategy="median")
    return X_df.values.astype(np.float64), y

def _load_synthetic(generator, params, random_state=42):
    from sklearn.datasets import (
        make_blobs, make_circles, make_classification, make_moons,
    )
    _generators = {
        "blobs":          make_blobs,
        "moons":          make_moons,
        "circles":        make_circles,
        "classification": make_classification,
        "xor":            _make_xor,
        "checkerboard":   _make_checkerboard,
        "two_spirals":    _make_two_spirals,
        "nonlin_subspace":_make_nonlin_subspace,
        "moons_rotated":  _make_moons_rotated,
    }
    if generator not in _generators:
        raise ValueError(f"Unknown generator: {generator!r}")
    X, y = _generators[generator](**params, random_state=random_state)
    return X.astype(np.float64), y.astype(int)

def load_classification_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load all binary classification datasets, subsampled to MAX_SAMPLES.

    Returns
    -------
    dict[str, tuple[ndarray, ndarray]]
        X : ndarray shape (n_samples, n_features), float64, RAW (unscaled)
            n_samples <= MAX_SAMPLES
        y : ndarray shape (n_samples,), int {0, 1}
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
            loader_fn = getattr(skd, spec["loader"])
            bunch = loader_fn(as_frame=True)
            if "classes" in spec:
                mask = bunch.target.isin(spec["classes"])
                X = bunch.data[mask].reset_index(drop=True).values.astype(float)
                y = bunch.target[mask].reset_index(drop=True)
                y = y.map({spec["classes"][0]: 0, spec["classes"][1]: 1}).values.astype(int)
            else:
                X = bunch.data.values.astype(float)
                y = bunch.target.values.astype(int)
            datasets[name] = _stratified_subsample(X, y)
        except Exception as e:
            logger.warning("Classification [sklearn] %s: %s", name, e)
            skipped.append(name)

    # ── 2. UCI ───────────────────────────────────────────────────────────────
    for name, spec in _UCI_DATASETS.items():
        spec = spec.copy()
        data_id = spec.pop("data_id")
        try:
            X, y = _load_uci(data_id, **spec)
            datasets[name] = _stratified_subsample(X, y)
        except Exception as e:
            logger.warning("Classification [uci] %s (id=%d): %s", name, data_id, e)
            skipped.append(name)

    # ── 3. CSV ───────────────────────────────────────────────────────────────
    for name, spec in _CSV_DATASETS.items():
        spec = spec.copy()
        path = DATA_DIR / spec.pop("path")
        if not path.exists():
            logger.warning("Classification [csv] %s: file not found: %s", name, path)
            skipped.append(name)
            continue
        try:
            X, y = _load_csv(path, **spec)
            datasets[name] = _stratified_subsample(X, y)
        except Exception as e:
            logger.warning("Classification [csv] %s: %s", name, e)
            skipped.append(name)

    # ── 4. Synthetic ─────────────────────────────────────────────────────────
    for name, spec in _SYNTHETIC_DATASETS.items():
        try:
            X, y = _load_synthetic(
                generator=spec["generator"],
                params=spec["params"],
            )
            datasets[name] = _stratified_subsample(X, y)
        except Exception as e:
            logger.warning("Classification [synthetic] %s: %s", name, e)
            skipped.append(name)

    logger.info(
        "Classification: loaded %d datasets%s",
        len(datasets),
        f", skipped {len(skipped)}: {skipped}" if skipped else "",
    )
    save_cache(datasets)
    return datasets