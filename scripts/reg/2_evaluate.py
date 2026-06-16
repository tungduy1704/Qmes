"""scripts/reg/2_evaluate.py"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from Qmes.data.reg.train import load_regression_datasets
from Qmes.evaluators.regression import RegressionEvaluator
from Qmes.circuits.registry import get_circuit_names

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("run_oracle_reg")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pivot_r2_regression_600samples_2nd.csv"

def main():
    datasets = load_regression_datasets()
    evaluator = RegressionEvaluator(n_splits=3, max_features=4, random_state=42)
    circuits = get_circuit_names()

    if OUT.exists():
        pivot = pd.read_csv(OUT, index_col=0)
        done = set(pivot.columns)
        logger.info("Resume: %d datasets đã xong", len(done))
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        pivot = pd.DataFrame(index=circuits)
        pivot.index.name = "circuit"
        done = set()

    for i, (name, (X, y)) in enumerate(datasets.items(), 1):
        if name in done:
            continue
        logger.info("[%d/%d] %s %s", i, len(datasets), name, X.shape)
        scores = evaluator.evaluate_all(X, y, circuits)
        pivot[name] = [scores[c].get("mean_r2", np.nan) for c in circuits]
        pivot.to_csv(OUT)  

    # ── Checkpoint ───────────────────────────────────────────────────────────
    print("\n=== CHECKPOINT ===")
    print(f"Pivot shape: {pivot.shape}  (circuits={len(circuits)}, datasets={len(datasets)})")
    nan_count = int(pivot.isna().sum().sum())
    print(f"NaN entries: {nan_count}")
    if nan_count:
        bad = pivot.columns[pivot.isna().any()].tolist()
        print(f"  ⚠ Cột có NaN: {bad}")

    clean = pivot.dropna(axis=1, how="all")
    print("\n=== Circuit ranking (mean R² across datasets) ===")
    print(clean.mean(axis=1).sort_values(ascending=False).round(4).to_string())
    print("\n=== Best-circuit count ===")
    print(clean.idxmax(axis=0).value_counts().to_string())

if __name__ == "__main__":
    main()