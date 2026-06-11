from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Qmes.data.preprocessing import encode_categoricals, impute_and_cast

logger = logging.getLogger(__name__)

DATA_DIR = Path(
    os.environ.get(
        "QMATCH_DATA_DIR",
        "qmatch/data complexity visualization/data_source",
    )
)

_SKLEARN_DATASETS: dict[str, dict] = {
    "Iris": {"loader": "load_iris"},
    "Wine": {"loader": "load_wine"},
    "Breast Cancer": {"loader": "load_breast_cancer"},
}

_UCI_DATASETS: dict[str, dict] = {
    "Obesity Levels": {
        "id": 544,
        "drop_cols": ["NObeyesdad"],   # target — bỏ để unsupervised
    },
    # 3 clusters hình cầu, benchmark cổ điển
    "Dry Bean": {
        "id": 602,
        "drop_cols": ["Class"],
    },
    # Phân cụm khách hàng ngân hàng (8 features numeric)
    "Bank Marketing": {
        "id": 222,
        "drop_cols": ["y"],            # target binary yes/no
    },
    # Đặc trưng âm thanh nhạc cụ, 10 classes → clustering thú vị
    "MAGIC Gamma Telescope": {
        "id": 159,
        "drop_cols": ["class"],
    },
    # Dữ liệu hạt giống lúa mì, 3 giống
    "Rice (Cammeo and Osmancik)": {
        "id": 545,
        "drop_cols": ["Class"],
    },
    # Chất lượng rượu vang đỏ — continuous features, phân cụm theo quality
    "Wine Quality Red": {
        "id": 186,
        "drop_cols": ["quality"],
    },
}

_SYNTHETIC_DATASETS: dict[str, dict] = {
    # ── Isotropic blobs (dễ cluster) ─────────────────────────────────────────
    "Blobs_2D_3C": {
        "generator": "make_blobs",
        "params": {"n_samples": 300, "n_features": 2, "centers": 3},
        "seed": 42,
    },
    "Blobs_5D_4C": {
        "generator": "make_blobs",
        "params": {"n_samples": 400, "n_features": 5, "centers": 4},
        "seed": 42,
    },
    "Blobs_10D_5C": {
        "generator": "make_blobs",
        "params": {"n_samples": 500, "n_features": 10, "centers": 5},
        "seed": 42,
    },
    # ── Anisotropic / elongated clusters (trung bình) ─────────────────────────
    "Blobs_Aniso_2D": {
        "generator": "make_blobs",
        "params": {
            "n_samples": 300,
            "n_features": 2,
            "centers": 3,
            "cluster_std": [1.0, 2.5, 0.5],   # std khác nhau → non-spherical
        },
        "seed": 7,
    },
    # ── Non-convex (khó cluster với k-means) ─────────────────────────────────
    "Moons_Noise01": {
        "generator": "make_moons",
        "params": {"n_samples": 300, "noise": 0.1},
        "seed": 42,
    },
    "Moons_Noise05": {
        "generator": "make_moons",
        "params": {"n_samples": 300, "noise": 0.5},   # noisy → harder
        "seed": 42,
    },
    "Circles_Factor02": {
        "generator": "make_circles",
        "params": {"n_samples": 300, "noise": 0.05, "factor": 0.3},
        "seed": 42,
    },
    # ── Manifold (non-linear structure) ──────────────────────────────────────
    "Swiss Roll": {
        "generator": "make_swiss_roll",
        "params": {"n_samples": 400, "noise": 0.1},
        "seed": 42,
    },
    "S Curve": {
        "generator": "make_s_curve",
        "params": {"n_samples": 400, "noise": 0.1},
        "seed": 42,
    },
    # ── High-dimensional (kiểm tra curse of dimensionality) ──────────────────
    "Blobs_20D_4C": {
        "generator": "make_blobs",
        "params": {"n_samples": 400, "n_features": 20, "centers": 4},
        "seed": 42,
    },
    "Gaussian Quantiles_3D": {
        "generator": "make_gaussian_quantiles",
        "params": {"n_samples": 300, "n_features": 3, "n_classes": 3},
        "seed": 42,
    },
}

"""_CSV_DATASETS: dict[str, dict] = {
    "Mall Customers": {
        "path": "Mall_Customers.csv",
        "drop_cols": ["CustomerID"],
    },
    "Seeds": {
        "path": "seeds_dataset.txt",
        "sep": r"\s+",
        "header": None,
        "columns": [
            "area", "perimeter", "compactness",
            "kernel_length", "kernel_width",
            "asymmetry", "groove_length", "label",
        ],
        "drop_cols": ["label"],
    },
    "Wholesale": {
        "path": "Wholesale customers data.csv",
        "drop_cols": ["Channel", "Region"],
    },
}"""

_CSV_DATASETS: dict[str, dict] = {}

def _df_to_array(df: pd.DataFrame) -> np.ndarray:
    """Encode categoricals → impute → cast float64."""
    df = encode_categoricals(df)
    df = impute_and_cast(df, strategy="median")
    return df.values.astype(np.float64)


def _load_sklearn_bunch(loader_name: str) -> np.ndarray:
    import sklearn.datasets as skd
    loader_fn = getattr(skd, loader_name)
    bunch = loader_fn(as_frame=True)
    return _df_to_array(bunch.data)


def _load_uci(
    uci_id: int,
    drop_cols: list[str] | None = None,
) -> np.ndarray:
    """Fetch một UCI dataset theo id, trả X ndarray."""
    from ucimlrepo import fetch_ucirepo

    repo = fetch_ucirepo(id=uci_id)
    X_df: pd.DataFrame = repo.data.features.copy()

    if drop_cols:
        existing = [c for c in drop_cols if c in X_df.columns]
        X_df = X_df.drop(columns=existing)

    return _df_to_array(X_df)


def _load_synthetic(
    generator_name: str,
    params: dict[str, Any],
    seed: int = 42,
) -> np.ndarray:
    """Gọi sklearn generator, trả X ndarray (bỏ y nếu có)."""
    import sklearn.datasets as skd

    gen_fn = getattr(skd, generator_name)

    # Thêm random_state nếu generator hỗ trợ
    import inspect
    sig = inspect.signature(gen_fn)
    kwargs = dict(params)
    if "random_state" in sig.parameters:
        kwargs["random_state"] = seed

    result = gen_fn(**kwargs)

    # Một số generators trả (X, y), một số chỉ trả X
    if isinstance(result, tuple):
        X = result[0]
    else:
        X = result

    return np.asarray(X, dtype=np.float64)


def _load_csv_clustering(
    path: Path,
    sep: str = ",",
    header: int | None = 0,
    columns: list[str] | None = None,
    drop_cols: list[str] | None = None,
) -> np.ndarray:
    df = pd.read_csv(path, sep=sep, header=header)
    if columns is not None:
        df.columns = columns
    if drop_cols is not None:
        existing = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=existing)
    return _df_to_array(df)

def load_clustering_datasets(
    include_sklearn: bool = True,
    include_uci: bool = True,
    include_synthetic: bool = True,
    include_csv: bool = False,
) -> dict[str, np.ndarray]:

    datasets: dict[str, np.ndarray] = {}
    skipped: list[str] = []

    # ── 1. sklearn bunches ───────────────────────────────────────────────────
    if include_sklearn:
        for name, spec in _SKLEARN_DATASETS.items():
            try:
                datasets[name] = _load_sklearn_bunch(spec["loader"])
            except Exception as e:
                logger.warning("Clustering [sklearn] %s: %s", name, e)
                skipped.append(name)

    # ── 2. UCI ───────────────────────────────────────────────────────────────
    if include_uci:
        try:
            from ucimlrepo import fetch_ucirepo  # noqa: F401 — availability check
        except ImportError:
            logger.warning(
                "Clustering [UCI]: ucimlrepo not installed. "
                "Run `pip install ucimlrepo` to enable UCI datasets."
            )
        else:
            for name, spec in _UCI_DATASETS.items():
                try:
                    datasets[name] = _load_uci(
                        uci_id=spec["id"],
                        drop_cols=spec.get("drop_cols"),
                    )
                except Exception as e:
                    logger.warning("Clustering [UCI] %s: %s", name, e)
                    skipped.append(name)

    # ── 3. Synthetic ─────────────────────────────────────────────────────────
    if include_synthetic:
        for name, spec in _SYNTHETIC_DATASETS.items():
            try:
                datasets[name] = _load_synthetic(
                    generator_name=spec["generator"],
                    params=spec.get("params", {}),
                    seed=spec.get("seed", 42),
                )
            except Exception as e:
                logger.warning("Clustering [synthetic] %s: %s", name, e)
                skipped.append(name)

    # ── 4. CSV ───────────────────────────────────────────────────────────────
    if include_csv:
        for name, spec in _CSV_DATASETS.items():
            spec = spec.copy()
            path = DATA_DIR / spec.pop("path")
            if not path.exists():
                logger.warning("Clustering [CSV] file not found: %s", path)
                skipped.append(name)
                continue
            try:
                datasets[name] = _load_csv_clustering(path, **spec)
            except Exception as e:
                logger.warning("Clustering [CSV] %s: %s", name, e)
                skipped.append(name)

    logger.info(
        "Clustering: loaded %d datasets%s",
        len(datasets),
        f", skipped {len(skipped)}: {skipped}" if skipped else "",
    )
    return datasets