
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _get_adbench_dir() -> Path:
    """Locate ADBench Classical datasets directory."""
    import adbench

    return Path(os.path.dirname(adbench.__file__)) / "datasets" / "Classical"

_ADBENCH_DATASETS: dict[str, tuple[str, int, int, float]] = {
    # name                  filename                      n_feat  n_samples  %anom
    "annthyroid":       ("2_annthyroid.npz",            6,   7200,   7.42),
    "breastw":          ("4_breastw.npz",               9,    683,  34.99),
    "cardio":           ("6_cardio.npz",               21,   1831,   9.61),
    "Cardiotocography": ("7_Cardiotocography.npz",     21,   2114,  22.04),
    "fault":            ("12_fault.npz",               27,   1941,  34.67),
    "glass":            ("14_glass.npz",                7,    214,   4.21),
    "Hepatitis":        ("15_Hepatitis.npz",           19,     80,  16.25),  # ⚠️ tiny
    "Ionosphere":       ("18_Ionosphere.npz",          32,    351,  35.90),  # ⚠️ borderline feat
    "Lymphography":     ("21_Lymphography.npz",        18,    148,   4.05),
    "mammography":      ("23_mammography.npz",          6,  11183,   2.32),
    "PageBlocks":       ("27_PageBlocks.npz",          10,   5393,   9.46),
    "pendigits":        ("28_pendigits.npz",           16,   6870,   2.27),
    "Pima":             ("29_Pima.npz",                 8,    768,  34.90),
    "satellite":        ("30_satellite.npz",           36,   6435,  31.64),  # ⚠️ borderline feat
    "satimage-2":       ("31_satimage-2.npz",          36,   5803,   1.22),  # ⚠️ borderline feat
    "Stamps":           ("37_Stamps.npz",               9,    340,   9.12),
    "thyroid":          ("38_thyroid.npz",              6,   3772,   2.47),
    "vertebral":        ("39_vertebral.npz",            6,    240,  12.50),
    "vowels":           ("40_vowels.npz",              12,   1456,   3.43),
    "Waveform":         ("41_Waveform.npz",            21,   3443,   2.90),
    "WBC":              ("42_WBC.npz",                  9,    223,   4.48),
    "WDBC":             ("43_WDBC.npz",                30,    367,   2.72),  # ⚠️ borderline feat
    "Wilt":             ("44_Wilt.npz",                 5,   4819,   5.33),
    "wine":             ("45_wine.npz",                13,    129,   7.75),  # ⚠️ tiny
    "WPBC":             ("46_WPBC.npz",                33,    198,  23.74),  # ⚠️ borderline feat
    "yeast":            ("47_yeast.npz",                8,   1484,  34.16),
}


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one ADBench .npz file.

    Returns
    -------
    X : ndarray shape (n_samples, n_features), float64
    y : ndarray shape (n_samples,), int {0, 1}
    """
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(int)

    # Handle NaN/Inf
    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())
    if nan_count > 0:
        logger.warning("%s: %d NaN in X, replacing with column median", path.name, nan_count)
        from sklearn.impute import SimpleImputer
        X = SimpleImputer(strategy="median").fit_transform(X)
    if inf_count > 0:
        logger.warning("%s: %d Inf in X, clipping", path.name, inf_count)
        X = np.clip(X, -1e10, 1e10)

    return X, y


def load_anomaly_detection_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load tất cả anomaly detection datasets đủ điều kiện.

    Returns
    -------
    dict[str, (X, y)]
        X : ndarray shape (n_samples, n_features), float64, RAW (chưa scale)
        y : ndarray shape (n_samples,), int  {0=normal, 1=anomaly}
    """
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    skipped: list[str] = []

    try:
        base_dir = _get_adbench_dir()
    except ImportError:
        logger.error("AnomalyDetection: adbench not installed. Run: pip install adbench")
        return datasets

    if not base_dir.exists():
        logger.error(
            "AnomalyDetection: dataset dir not found: %s\n"
            "  Run: from adbench.myutils import Utils; Utils().download_datasets(repo='jihulab')",
            base_dir,
        )
        return datasets

    for name, (filename, *_) in _ADBENCH_DATASETS.items():
        path = base_dir / filename
        if not path.exists():
            logger.warning("AnomalyDetection: file not found — %s", path)
            skipped.append(name)
            continue
        try:
            datasets[name] = _load_npz(path)
        except Exception as e:
            logger.warning("AnomalyDetection: failed to load %s: %s", name, e)
            skipped.append(name)

    logger.info(
        "AnomalyDetection: loaded %d / %d datasets%s",
        len(datasets),
        len(_ADBENCH_DATASETS),
        f" | skipped {len(skipped)}: {skipped}" if skipped else "",
    )
    return datasets

