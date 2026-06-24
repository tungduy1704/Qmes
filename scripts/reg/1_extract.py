"""scripts/reg/1_extracts.py"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from Qmes.data.reg.train import load_regression_datasets
from Qmes.extractors.regression import RegressionExtractor

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("run_extractor_reg")

OUT = Path(__file__).resolve().parents[2] / "results" / \
    "meta_dataset_regression_single_avg_600samples_2nd.csv"

def main():
    datasets = load_regression_datasets()
    logger.info("Loaded %d datasets", len(datasets))

    extractor = RegressionExtractor()
    rows, failed = {}, []

    for i, (name, (X, y)) in enumerate(datasets.items(), 1):
        try:
            result = extractor.extract(X, y)
            rows[name] = np.asarray(result.vector, dtype=np.float64)
            logger.info("[%d/%d] %s  shape=%s  OK", i, len(datasets), name, X.shape)
        except Exception:
            logger.exception("[%d/%d] %s  FAILED", i, len(datasets), name)
            failed.append(name)

    feature_names = getattr(extractor, "_feature_names", None) \
        or [f"f{i}" for i in range(len(next(iter(rows.values()))))]
    df = pd.DataFrame.from_dict(rows, orient="index", columns=feature_names)
    df.index.name = "dataset"

    print("\n=== CHECKPOINT ===")
    print(f"Shape: {df.shape}  (expect ({len(datasets) - len(failed)}, 12))")
    print(f"Failed: {failed if failed else 'none'}")
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        print(f" NaN ở cột: {nan_cols}")
        print(df[df.isna().any(axis=1)].index.tolist())
    else:
        print("NaN: none")
    zero_rows = df.index[(df == 0).all(axis=1)].tolist()
    if zero_rows:
        print(f" {len(zero_rows)} all-zero row (extraction failures silently swallowed): {zero_rows}")
    else:
        print("Zero rows: none")

    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()