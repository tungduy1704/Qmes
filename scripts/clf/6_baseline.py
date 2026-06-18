"""scripts/clf/6_baseline.py

Baseline comparison for the classification recommender.

Evaluates whether the meta-learning recommender outperforms fixed-circuit
baselines on per-dataset regret (MCC), using LOO-fair evaluation (n=105).

Baselines
---------
LOO best-average  : circuit with highest mean MCC across the remaining n-1
                    datasets — the strongest fair baseline, and the primary
                    bar the recommender must beat.
LOO modal-best    : circuit most frequently ranked best across n-1 datasets.
random            : expected regret of uniform random circuit selection.
best fixed (oracle) : lowest mean regret when all labels are known (invalid
                    in practice; reported for reference only).

Statistical test
----------------
Wilcoxon signed-rank on paired per-dataset regret vectors
(recommender vs LOO best-average) + win/loss/tie counts.

Usage
-----
    python scripts/clf/6_baseline.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from Qmes.config import TIED_THRESHOLD
from Qmes.evaluators.classification import filter_degenerate_datasets
from Qmes.recommender.selection import DEFAULT_CLASSIFIERS, select_features_mi

from diagnostic_top1_stratified import loo_per_dataset

ROOT = Path(__file__).resolve().parents[2]
META_PATH = ROOT / "results" / "meta_dataset_classification_single_avg_600samples.csv"
PIVOT_PATH = ROOT / "results" / "pivot_mcc_classification_600samples.csv"

K_VALUES = [5, 10, 15, 20]
CANDIDATES = [("kNN", "top5"), ("kNN", "top10")]


def baseline_regret_table(pivot):
    circuits = list(pivot.index)
    datasets = list(pivot.columns)
    best = pivot.max(axis=0) 

    rows = []
    for ds in datasets:
        others = [d for d in datasets if d != ds]
        sub = pivot[others]
        c_avg = sub.mean(axis=1).idxmax()                 
        c_modal = sub.idxmax(axis=0).value_counts().idxmax()  
        rows.append({
            "dataset":     ds,
            "reg_bestavg": float(best[ds] - pivot.loc[c_avg, ds]),
            "reg_modal":   float(best[ds] - pivot.loc[c_modal, ds]),
            "reg_random":  float((best[ds] - pivot[ds]).mean()),  
        })
    df = pd.DataFrame(rows)

    fixed = {c: float((best - pivot.loc[c]).mean()) for c in circuits}  
    return df, fixed

def main():
    meta = pd.read_csv(META_PATH, index_col=0)
    pivot = pd.read_csv(PIVOT_PATH, index_col=0)
    pivot_f, _ = filter_degenerate_datasets(
        pivot, min_max_score=0.1, ceiling_threshold=0.99
    )
    meta_f = meta.loc[pivot_f.columns]
    X_all = meta_f.values
    y_single = pivot_f.idxmax(axis=0).values
    subsets = select_features_mi(X_all, y_single, k_values=K_VALUES)

    base_df, fixed = baseline_regret_table(pivot_f)
    n = pivot_f.shape[1]

    print(f"=== Baseline mean regret (n={n}, LOO-fair) ===")
    print(f"  LOO best-average circuit : {base_df.reg_bestavg.mean():.4f}   <-- mốc phải vượt")
    print(f"  LOO modal-best circuit   : {base_df.reg_modal.mean():.4f}")
    print(f"  Random circuit           : {base_df.reg_random.mean():.4f}")
    print("  best fixed in hindsight (CHEAT, chỉ tham chiếu):")
    for c, r in sorted(fixed.items(), key=lambda kv: kv[1]):
        print(f"      {c:<6}: {r:.4f}")
    print()

    for clf_key, subset in CANDIDATES:
        clf = DEFAULT_CLASSIFIERS[clf_key]
        rec = loo_per_dataset(X_all, pivot_f, clf, subsets[subset],
                              tied_threshold=TIED_THRESHOLD)
        merged = rec[["dataset", "regret"]].merge(base_df, on="dataset")

        rec_mean = merged.regret.mean()
        base_mean = merged.reg_bestavg.mean()
        diff = merged.reg_bestavg - merged.regret  
        try:
            _, p = wilcoxon(merged.regret, merged.reg_bestavg)
        except ValueError:
            p = float("nan")

        print(f"=== {clf_key}_{subset} ===")
        print(f"  recommender regret       : {rec_mean:.4f}")
        print(f"  LOO best-average regret  : {base_mean:.4f}")
        print(f"  lift (base - rec)        : {base_mean - rec_mean:+.4f}")
        print(f"  Wilcoxon p (paired)      : {p:.4f}")
        print(f"  rec better/worse/tie     : "
              f"{int((diff > 0).sum())}/{int((diff < 0).sum())}/{int((diff == 0).sum())}\n")

if __name__ == "__main__":
    main()