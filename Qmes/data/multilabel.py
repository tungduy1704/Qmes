"""qmatch/data/multilabel.py"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import arff
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

_MULAN_NS = {"mulan": "http://mulan.sourceforge.net/labels"}


def _parse_label_names(xml_path: Path) -> list[str]:
    """Extract label names from Mulan XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = [
        label.get("name")
        for label in root.findall("mulan:label", _MULAN_NS)
    ]
    if not labels:
        # Fallback: some XML files don't use namespace
        labels = [label.get("name") for label in root.findall("label")]
    return labels


def _load_arff(path: Path) -> pd.DataFrame:
    """Load single ARFF file into DataFrame (liac-arff)."""
    with open(path, "r") as f:
        data = arff.load(f)
    return pd.DataFrame(
        data["data"],
        columns=[attr[0] for attr in data["attributes"]],
    )


def _load_mulan_dataset(
    folder: Path,
    name: str,
    drop_cols: list[str] | None = None,
    encode_map: dict[str, dict] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one Mulan multi-label dataset from folder.

    Parameters
    ----------
    folder : Path containing {name}-train.arff, {name}-test.arff, {name}.xml
    name : dataset name (used to find files)
    drop_cols : columns to drop from features (e.g. ID columns)
    encode_map : manual encoding for specific columns
        e.g. {"protein": {"NO": 0, "YES": 1}}

    Returns
    -------
    X : ndarray shape (n_samples, n_features), float64
    Y : ndarray shape (n_samples, n_labels), int {0, 1}
    """
    xml_path = folder / f"{name}.xml"
    train_path = folder / f"{name}-train.arff"
    test_path = folder / f"{name}-test.arff"

    # Parse label names from XML
    label_names = _parse_label_names(xml_path)

    # Load and concat train + test
    train = _load_arff(train_path)
    test = _load_arff(test_path)
    df = pd.concat([train, test], ignore_index=True)

    # Split into label columns and feature columns
    label_cols = df.columns[-len(label_names) :].tolist()
    Y = df[label_cols].astype(int).values

    X_df = df.drop(columns=label_cols)

    # Drop specified columns (e.g. ID columns)
    if drop_cols is not None:
        existing = [c for c in drop_cols if c in X_df.columns]
        X_df = X_df.drop(columns=existing)

    # Manual encoding for specific columns (e.g. NO/YES → 0/1)
    if encode_map is not None:
        for col, mapping in encode_map.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].map(mapping)

    # Standard preprocessing
    X_df = encode_categoricals(X_df)
    X_df = impute_and_cast(X_df, strategy="median")
    X = X_df.values.astype(np.float64)

    return X, Y


# ── Dataset registry ────────────────────────────────────────────────────────
_MULAN_DATASETS: dict[str, dict] = {
    "Birds": {"folder": "birds", "name": "birds"},
    "Emotions": {"folder": "emotions", "name": "emotions"},
    "Flags": {"folder": "flags", "name": "flags"},
    "Genbase": {
        "folder": "genbase",
        "name": "genbase",
        "drop_cols": ["protein"],
        "encode_map": {
            col: {"NO": 0, "YES": 1}
            for col in []  # encode_categoricals sẽ handle tất cả
        },
    },
    "Medical": {"folder": "medical", "name": "medical"},
}


def load_multilabel_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load multi-label classification datasets.

    Returns
    -------
    dict[str, (X, Y)]
        X : ndarray shape (n_samples, n_features), float64, RAW (chưa scale)
        Y : ndarray shape (n_samples, n_labels), int {0, 1}
    """
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    skipped: list[str] = []

    for display_name, spec in _MULAN_DATASETS.items():
        spec = spec.copy()
        folder = DATA_DIR / spec.pop("folder")

        if not folder.exists():
            logger.warning("Multi-label: folder not found: %s", folder)
            skipped.append(display_name)
            continue

        try:
            datasets[display_name] = _load_mulan_dataset(folder, **spec)
        except Exception as e:
            logger.warning(
                "Multi-label: failed to load %s: %s", display_name, e
            )
            skipped.append(display_name)

    logger.info(
        "Multi-label: loaded %d datasets%s",
        len(datasets),
        f", skipped {len(skipped)}: {skipped}" if skipped else "",
    )

    return datasets