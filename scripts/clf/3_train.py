"""scripts/clf/3_train.py"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from Qmes.config import TIED_THRESHOLD
from Qmes.evaluators.classification import filter_degenerate_datasets
from Qmes.recommender.selection import (
    DEFAULT_CLASSIFIERS,
    select_features_mi,
    run_loo_evaluation,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[2]
META_PATH = ROOT / "results" / "meta_dataset_classification_single_avg_600samples.csv"
PIVOT_PATH = ROOT / "results" / "pivot_mcc_classification_600samples.csv"
OUT_PATH = ROOT / "results" / "recommender_clf_summary_600samples.csv"

K_VALUES = [5, 10, 15, 20]

SMOKE = False  # quick test: 2 classifiers × 1 subset; set to False for full run

def main():
    meta = pd.read_csv(META_PATH, index_col=0)
    pivot = pd.read_csv(PIVOT_PATH, index_col=0)

    common = [ds for ds in pivot.columns if ds in meta.index]
    assert len(common) == len(pivot.columns) == len(meta.index), \
        f"Mismatch: pivot={len(pivot.columns)}, meta={len(meta.index)}, common={len(common)}"

    pivot_f, reasons = filter_degenerate_datasets(
        pivot, min_max_score=0.1, ceiling_threshold=0.99
    )
    print(f"Filtered {len(reasons)} datasets:")
    for ds, why in reasons.items():
        print(f"  - {ds}: {why}")
    print(f"Remaining: {pivot_f.shape[1]} datasets\n")

    datasets = list(pivot_f.columns)
    meta_f = meta.loc[pivot_f.columns]
    X_all = meta_f.values
    feature_names = list(meta_f.columns)

    tied_sets = {
        ds: set(pivot_f[ds][pivot_f[ds] >= pivot_f[ds].max() - TIED_THRESHOLD].index)
        for ds in datasets
    }
    n = len(datasets)
    print("=== Baselines (sau filter) ===")
    print(f"Mean tie size: {np.mean([len(s) for s in tied_sets.values()]):.2f}")
    for c in pivot_f.index:
        tied_acc = sum(c in tied_sets[ds] for ds in datasets) / n
        print(f"  Always '{c}': Tied={tied_acc:.4f}")
        from math import comb
        p_rand_top3 = np.mean([
            1 - comb(7 - len(tied_sets[ds]), 3) / comb(7, 3) if len(tied_sets[ds]) <= 4 else 1.0
            for ds in datasets
        ])
        print(f"  Random top-3: Top3_Tied≈{p_rand_top3:.4f}")
    print()

    # ── MI feature selection ──────────────────────────────────
    y_single = pivot_f.idxmax(axis=0).values
    feature_subsets = select_features_mi(X_all, y_single, k_values=K_VALUES)

    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(X_all, y_single, random_state=42)
    print("=== MI Feature Ranking ===")
    for rank, idx in enumerate(np.argsort(mi)[::-1], 1):
        print(f"  {rank:>2}. {feature_names[idx]:<10} MI={mi[idx]:.4f}")
    print()

    # ── LOO grid ──────────────────────────────────────────────
    if SMOKE:
        classifiers = {k: DEFAULT_CLASSIFIERS[k] for k in ["GB", "NaiveBayes"]}
        feature_subsets = {"top10": feature_subsets["top10"]}
        print(">>> SMOKE MODE: 2 classifiers × 1 subset <<<\n")
    else:
        classifiers = DEFAULT_CLASSIFIERS

    summary = run_loo_evaluation(
        X_all, pivot_f,
        classifiers=classifiers,
        feature_subsets=feature_subsets,
        tied_threshold=TIED_THRESHOLD,
        verbose=True,
        dataset_names=list(meta_f.index),  # alignment guard: meta rows ↔ pivot cols
    )

    # ── Summary ───────────────────────────────────────────────
    print("\n=== Top 10 by Mean Regret ===")
    print(summary.sort_values("Mean_Regret").head(10).to_string(index=False))
    print("\n=== Top 10 by Tied ===")
    print(summary.sort_values("Tied", ascending=False).head(10).to_string(index=False))
    safe = summary[summary["Top3_Tied"] >= 0.90]
    print(f"\n=== Safe configs (Top3_Tied >= 0.90): {len(safe)} ===")
    print(safe.sort_values("Mean_Regret").head(10).to_string(index=False))

    if not SMOKE:
        summary.to_csv(OUT_PATH, index=False)
        print(f"\nSaved: {OUT_PATH}")

if __name__ == "__main__":
    main()