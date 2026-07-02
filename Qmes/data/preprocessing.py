"""Qmes/data/preprocessing.py"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """LabelEncode for all object/category."""
    X = X.copy()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    return X

def impute_and_cast(X: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Impute missing, cast to float."""
    X = X.apply(pd.to_numeric, errors="coerce")
    arr = SimpleImputer(strategy=strategy).fit_transform(X)
    return pd.DataFrame(arr, columns=X.columns)

def scale_features(X: np.ndarray, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler

# ── Time series preprocessing (unused — reserved for future task type,
#    not part of the classification/regression scope described in the
#    software paper. Not imported anywhere in Qmes/, scripts/, or tests/.) ──
def impute_timeseries(X: np.ndarray, strategy: str = "zero") -> np.ndarray:
    """Impute NaN in time series 2D (n_samples, n_timesteps).

    Parameters
    ----------
    strategy : 'zero', 'forward', 'mean'
        - zero: replace NaN by 0
        - forward: forward-fill by timestep, first NaN fill by 0
        - mean: replace NaN by mean of each series
    """
    X = X.copy()
    if not np.isnan(X).any():
        return X

    if strategy == "zero":
        X = np.nan_to_num(X, nan=0.0)
    elif strategy == "forward":
        for i in range(X.shape[0]):
            series = pd.Series(X[i])
            series = series.ffill().fillna(0.0)
            X[i] = series.values
    elif strategy == "mean":
        for i in range(X.shape[0]):
            mean_val = np.nanmean(X[i])
            if np.isnan(mean_val):
                mean_val = 0.0
            X[i] = np.where(np.isnan(X[i]), mean_val, X[i])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return X


def normalize_timeseries(
    X: np.ndarray, method: str = "zscore"
) -> np.ndarray:
    """Normalize each series independently.

    Parameters
    ----------
    method : 'zscore', 'minmax'
        - zscore: (x - mean) / std per series (standard for catch22/tsfresh)
        - minmax: scale [0, 1] per series
    """
    X = X.copy().astype(np.float64)

    if method == "zscore":
        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-12, 1.0, stds)
        X = (X - means) / stds
    elif method == "minmax":
        mins = X.min(axis=1, keepdims=True)
        maxs = X.max(axis=1, keepdims=True)
        denom = maxs - mins
        denom = np.where(denom < 1e-12, 1.0, denom)
        X = (X - mins) / denom
    else:
        raise ValueError(f"Unknown method: {method}")

    return X