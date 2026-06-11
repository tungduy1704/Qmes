"""qmatch/data/timeseries.py"""

from __future__ import annotations
import logging
import numpy as np
logger = logging.getLogger(__name__)
from Qmes.data.preprocessing import impute_timeseries

_AEON_DATASETS: dict[str, dict] = {
    "GunPoint": {},
    "ECGFiveDays": {},
    "ItalyPowerDemand": {},
    "SonyAIBORobotSurface1": {},
    "ArrowHead": {},
    "CBF": {},
    "Beef": {},
}


def _load_aeon_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one UCR dataset via aeon, concat train+test, squeeze to 2D.

    Parameters
    ----------
    name : UCR dataset name (case-sensitive)

    Returns
    -------
    X : ndarray shape (n_samples, n_timesteps), float64
    y : ndarray shape (n_samples,), int starting from 0
    """
    from aeon.datasets import load_classification

    X_train, y_train = load_classification(name, split="train")
    X_test, y_test = load_classification(name, split="test")

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    # aeon trả (n_samples, n_channels, n_timesteps) — squeeze univariate
    if X.ndim == 3 and X.shape[1] == 1:
        X = X.squeeze(axis=1)

    X = X.astype(np.float64)

    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        logger.warning("%s: %d NaN values in X", name, nan_count)
        X = impute_timeseries(X, strategy="zero")

    # Re-encode y về int starting from 0
    # aeon trả y dạng string ("1", "2") hoặc số không liên tục
    classes = sorted(set(y))
    class_map = {c: i for i, c in enumerate(classes)}
    y = np.array([class_map[c] for c in y], dtype=int)

    return X, y


def load_timeseries_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load univariate time series classification datasets.

    Returns
    -------
    dict[str, (X, y)]
        X : ndarray shape (n_samples, n_timesteps), float64, RAW (chưa scale)
        y : ndarray shape (n_samples,), int starting from 0
    """
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    skipped: list[str] = []

    for name, spec in _AEON_DATASETS.items():
        try:
            datasets[name] = _load_aeon_dataset(name)
        except Exception as e:
            logger.warning("TimeSeries: failed to load %s: %s", name, e)
            skipped.append(name)

    logger.info(
        "TimeSeries: loaded %d datasets%s",
        len(datasets),
        f", skipped {len(skipped)}: {skipped}" if skipped else "",
    )

    return datasets