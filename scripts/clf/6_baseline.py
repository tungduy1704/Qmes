"""scripts/clf/6_baseline.py

Baseline comparison for the classification recommender.

Evaluates whether the meta-learning recommender outperforms fixed-circuit
baselines on per-dataset regret (MCC), using LOO-fair evaluation.

Baselines
---------
LOO best-average : circuit with highest mean MCC across the remaining n-1
                   datasets — strongest fair baseline, primary bar to beat.
LOO modal-best   : circuit most frequently ranked best across n-1 datasets.
random           : expected regret of uniform random circuit selection.

Statistical test
----------------
Wilcoxon signed-rank on paired per-dataset regret vectors
(recommender vs LOO best-average) + win/loss/tie counts.

Reported candidate
------------------
kNN_top10 — best config by LOO Mean_Regret (see recommender_clf_summary).

Usage
-----
    python scripts/clf/6_baseline.py
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

from Qmes.config import TIED_THRESHOLD
from Qmes.evaluators.base import filter_degenerate_datasets
from Qmes.recommender.selection import select_features_mi

ROOT       = Path(__file__).resolve().parents[2]
META_PATH  = ROOT / "results" / "meta_dataset_classification_single_avg_600samples.csv"
PIVOT_PATH = ROOT / "results" / "pivot_mcc_classification_600samples.csv"

# Best config selected by LOO Mean_Regret (recommender_clf_summary_600samples.csv)
BEST_CLF    = KNeighborsClassifier()
BEST_SUBSET = "top10"
K_VALUES    = [5, 10, 15, 20]


# ── LOO per-dataset ───────────────────────────────────────────────────────────

def loo_per_dataset(
    meta_features: np.ndarray,
    pivot: pd.DataFrame,
    classifier,
    feat_idx: list[int],
    tied_threshold: float = TIED_THRESHOLD,
) -> pd.DataFrame:
    """Run LOO evaluation, return per-dataset regret and hit metrics."""
    circuits  = list(pivot.index)
    datasets  = list(pivot.columns)
    pairs     = list(combinations(circuits, 2))
    y_single  = pivot.idxmax(axis=0).values
    tied_sets = {
        ds: set(pivot[ds][pivot[ds] >= pivot[ds].max() - tied_threshold].index)
        for ds in datasets
    }
    pairwise_y = {
        (c1, c2): np.array([
            1 if pivot.loc[c1, ds] >= pivot.loc[c2, ds] else 0
            for ds in datasets
        ])
        for (c1, c2) in pairs
    }

    X   = meta_features[:, feat_idx]
    loo = LeaveOneOut()
    rows = []

    for train_idx, test_idx in loo.split(meta_features):
        i  = test_idx[0]
        ds = datasets[i]

        votes = {c: 0 for c in circuits}
        for (c1, c2) in pairs:
            clf = clone(classifier)
            clf.fit(X[train_idx], pairwise_y[(c1, c2)][train_idx])
            if clf.predict(X[test_idx])[0] == 1:
                votes[c1] += 1
            else:
                votes[c2] += 1

        mv_pred = max(votes, key=votes.get)
        ranked  = sorted(votes, key=votes.get, reverse=True)[:3]

        rows.append({
            "dataset":    ds,
            "true_best":  y_single[i],
            "pred":       mv_pred,
            "tied_hit":   int(mv_pred in tied_sets[ds]),
            "top3_hit":   int(any(c in tied_sets[ds] for c in ranked)),
            "regret":     float(max(0.0, pivot[ds].max() - pivot.loc[mv_pred, ds])),
        })

    return pd.DataFrame(rows)


# ── Baseline table ────────────────────────────────────────────────────────────

def baseline_regret_table(pivot: pd.DataFrame) -> pd.DataFrame:
    """Compute per-dataset regret for each LOO-fair baseline."""
    datasets = list(pivot.columns)
    best     = pivot.max(axis=0)

    rows = []
    for ds in datasets:
        others  = [d for d in datasets if d != ds]
        sub     = pivot[others]
        c_avg   = sub.mean(axis=1).idxmax()
        c_modal = sub.idxmax(axis=0).value_counts().idxmax()
        rows.append({
            "dataset":     ds,
            "reg_bestavg": float(best[ds] - pivot.loc[c_avg,   ds]),
            "reg_modal":   float(best[ds] - pivot.loc[c_modal, ds]),
            "reg_random":  float((best[ds] - pivot[ds]).mean()),
        })

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    meta  = pd.read_csv(META_PATH,  index_col=0)
    pivot = pd.read_csv(PIVOT_PATH, index_col=0)

    pivot_f, removed = filter_degenerate_datasets(
        pivot, min_max_score=0.1, ceiling_threshold=0.99
    )
    if removed:
        print(f"Removed {len(removed)} degenerate datasets: {list(removed.keys())}")

    meta_f = meta.loc[pivot_f.columns]
    X_all  = meta_f.values
    n      = pivot_f.shape[1]

    # Feature subsets via MI
    y_single = pivot_f.idxmax(axis=0).values
    subsets  = select_features_mi(X_all, y_single, k_values=K_VALUES)

    # Baselines
    base_df  = baseline_regret_table(pivot_f)

    print(f"\n=== Baselines (n={n}, LOO-fair) ===")
    print(f"  LOO best-average : {base_df.reg_bestavg.mean():.4f}  <-- bar to beat")
    print(f"  LOO modal-best   : {base_df.reg_modal.mean():.4f}")
    print(f"  Random           : {base_df.reg_random.mean():.4f}")

    # Recommender (best config only)
    feat_idx = subsets[BEST_SUBSET]
    rec      = loo_per_dataset(X_all, pivot_f, BEST_CLF, feat_idx)
    merged   = rec[["dataset", "regret"]].merge(base_df, on="dataset")

    rec_mean  = merged.regret.mean()
    base_mean = merged.reg_bestavg.mean()
    diff      = merged.reg_bestavg - merged.regret

    try:
        _, p = wilcoxon(merged.regret, merged.reg_bestavg)
    except ValueError:
        p = float("nan")

    print(f"\n=== kNN_{BEST_SUBSET} (best config) ===")
    print(f"  Recommender Mean Regret  : {rec_mean:.4f}")
    print(f"  LOO best-average Regret  : {base_mean:.4f}")
    print(f"  Lift (base - rec)        : {base_mean - rec_mean:+.4f}")
    print(f"  Wilcoxon p (paired)      : {p:.4f}")
    print(f"  Win / Loss / Tie         : "
          f"{int((diff>0).sum())} / {int((diff<0).sum())} / {int((diff==0).sum())}")


if __name__ == "__main__":
    main()