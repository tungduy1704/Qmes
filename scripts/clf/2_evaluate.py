"""scripts/clf/2_evaluate.py"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from Qmes.data.clf.train import load_classification_datasets
from Qmes.evaluators.classification import ClassificationEvaluator
from Qmes.circuits.registry import get_circuit_names

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("run_oracle_clf")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "pivot_mcc_classification_600samples.csv"
BACKUP = ROOT / "backup" / "pivot_mcc_classification.csv"


def main():
    datasets = load_classification_datasets()
    evaluator = ClassificationEvaluator(n_splits=3, max_features=4, random_state=42)
    circuits = get_circuit_names()

    if OUT.exists():
        pivot = pd.read_csv(OUT, index_col=0)
        done = set(pivot.columns)
        logger.info("Resume: %d datasets done", len(done))
    else:
        pivot = pd.DataFrame(index=circuits)
        pivot.index.name = "circuit"
        done = set()

    for i, (name, (X, y)) in enumerate(datasets.items(), 1):
        if name in done:
            continue
        logger.info("[%d/%d] %s %s", i, len(datasets), name, X.shape)
        scores = evaluator.evaluate_all(X, y, circuits)
        pivot[name] = [scores[c].get("mean_mcc", np.nan) for c in circuits]
        pivot.to_csv(OUT)  

    print("\n=== CHECKPOINT ===")
    print(f"Pivot shape: {pivot.shape}  (expect (7, 108))")
    nan_count = int(pivot.isna().sum().sum())
    print(f"NaN entries: {nan_count}")

    if BACKUP.exists():
        old = pd.read_csv(BACKUP, index_col=0)
        uncapped = [n for n, (X, _) in datasets.items() if len(X) < 300]
        common = [n for n in uncapped if n in old.columns and n in pivot.columns]
        if common:
            diff = (pivot[common] - old.loc[pivot.index, common]).abs()
            print(f"Comparing {len(common)} non-capped datasets against backup:")
            print(f"  max |ΔMCC| = {diff.max().max():.6f}  (expect ≈ 0)")
            bad = diff.columns[(diff > 6e-5).any()].tolist()
            if bad:
                print(f" Unexpected mismatch at: {bad}")
        else:
            print("(No shared datasets to compare — check column names in backup)")
    else:
        print("(No backup found to compare against)")


if __name__ == "__main__":
    main()