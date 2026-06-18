"""scripts/reg/6_baseline.py

Baseline comparison for the regression recommender.

Evaluates whether the meta-learning recommender outperforms fixed-circuit
baselines on per-dataset regret (R²), using LOO-fair evaluation (n=88).

Baselines
---------
LOO best-average  : circuit with highest mean R² across the remaining n-1
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
    python scripts/reg/6_baseline.py
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from Qmes.config import TIED_THRESHOLD

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

META_PATH  = ROOT / "results" / "meta_dataset_regression_single_avg_600samples_2nd.csv"
PIVOT_PATH = ROOT / "results" / "pivot_r2_regression_600samples_2nd.csv"

DEGENERATE = [
    "Syn_Reg_sin_highdim",
    "Forest Fires",
    "Metro Interstate",
    "Syn_Reg_outlier_high",
    "Syn_Reg_friedman1_20f",
    "Syn_Reg_highdim_40f",
    "Syn_Reg_hetero_high",
    "Syn_Reg_mixed_sin_10f",
    "Appliances Energy",
    "Beijing PM2.5",
    "Bias Correction",
]

CANDIDATES = [
    ("kNN",        "top10"),
    ("NaiveBayes", "full"),
]

CLASSIFIERS = {
    "kNN":        KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
}

# ── LOO per-dataset ───────────────────────────────────────────────────────────
def loo_per_dataset(meta_features, pivot_r2, classifier, feat_idx,
                    tied_threshold=TIED_THRESHOLD):
    circuits  = list(pivot_r2.index)
    datasets  = list(pivot_r2.columns)
    pairs     = list(combinations(circuits, 2))
    y_single  = pivot_r2.idxmax(axis=0).values
    tied_sets = {
        ds: set(pivot_r2[ds][pivot_r2[ds] >= pivot_r2[ds].max() - tied_threshold].index)
        for ds in datasets
    }
    pairwise_y = {
        (c1, c2): np.array([
            1 if pivot_r2.loc[c1, ds] >= pivot_r2.loc[c2, ds] else 0
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
            "tie_size":   len(tied_sets[ds]),
            "true_best":  y_single[i],
            "pred":       mv_pred,
            "single_hit": int(mv_pred == y_single[i]),
            "tied_hit":   int(mv_pred in tied_sets[ds]),
            "top3_hit":   int(any(c in tied_sets[ds] for c in ranked)),
            "regret":     float(max(0.0, pivot_r2[ds].max() - pivot_r2.loc[mv_pred, ds])),
        })

    return pd.DataFrame(rows)

# ── Baseline table ────────────────────────────────────────────────────────────
def baseline_regret_table(pivot_r2):
    circuits = list(pivot_r2.index)
    datasets = list(pivot_r2.columns)
    best     = pivot_r2.max(axis=0)

    rows = []
    for ds in datasets:
        others  = [d for d in datasets if d != ds]
        sub     = pivot_r2[others]
        c_avg   = sub.mean(axis=1).idxmax()
        c_modal = sub.idxmax(axis=0).value_counts().idxmax()
        rows.append({
            "dataset":     ds,
            "reg_bestavg": float(best[ds] - pivot_r2.loc[c_avg, ds]),
            "reg_modal":   float(best[ds] - pivot_r2.loc[c_modal, ds]),
            "reg_random":  float((best[ds] - pivot_r2[ds]).mean()),
        })

    df    = pd.DataFrame(rows)
    fixed = {c: float((best - pivot_r2.loc[c]).mean()) for c in circuits}
    return df, fixed


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    meta  = pd.read_csv(META_PATH, index_col=0)
    meta  = meta.drop(columns=["meta_label"], errors="ignore")
    pivot = pd.read_csv(PIVOT_PATH, index_col=0)

    # Degenerate filter
    meta  = meta.drop(index=[d for d in DEGENERATE if d in meta.index])
    pivot = pivot.drop(columns=[d for d in DEGENERATE if d in pivot.columns],
                       errors="ignore")

    # Align
    common = [ds for ds in meta.index if ds in pivot.columns]
    meta   = meta.loc[common]
    pivot  = pivot[common]
    X_all  = meta.values
    n      = len(common)

    print(f"n datasets (sau filter): {n}")

    # MI feature subsets
    y_single   = pivot.idxmax(axis=0).values
    mi_scores  = mutual_info_classif(X_all, y_single, random_state=42)
    ranked_idx = np.argsort(mi_scores)[::-1]
    subsets    = {
        "full":  list(range(X_all.shape[1])),
        "top10": ranked_idx[:10].tolist(),
    }

    # Baselines
    base_df, fixed = baseline_regret_table(pivot)

    print(f"\n=== Baseline mean regret (n={n}, LOO-fair) ===")
    print(f"  LOO best-average circuit : {base_df.reg_bestavg.mean():.4f}   <-- mốc phải vượt")
    print(f"  LOO modal-best circuit   : {base_df.reg_modal.mean():.4f}")
    print(f"  Random circuit           : {base_df.reg_random.mean():.4f}")
    print("  best fixed in hindsight (CHEAT, chỉ tham chiếu):")
    for c, r in sorted(fixed.items(), key=lambda kv: kv[1]):
        print(f"      {c:<6}: {r:.4f}")
    print()

    # Per-candidate
    for clf_key, feat_label in CANDIDATES:
        clf      = CLASSIFIERS[clf_key]
        feat_idx = subsets[feat_label]

        rec    = loo_per_dataset(X_all, pivot, clf, feat_idx)
        merged = rec[["dataset", "regret"]].merge(base_df, on="dataset")

        rec_mean  = merged.regret.mean()
        base_mean = merged.reg_bestavg.mean()
        diff      = merged.reg_bestavg - merged.regret

        try:
            _, p = wilcoxon(merged.regret, merged.reg_bestavg)
        except ValueError:
            p = float("nan")

        better = int((diff > 0).sum())
        worse  = int((diff < 0).sum())
        tie_n  = int((diff == 0).sum())

        print(f"=== {clf_key}_{feat_label} ===")
        print(f"  recommender mean regret  : {rec_mean:.4f}")
        print(f"  LOO best-average regret  : {base_mean:.4f}")
        print(f"  lift (base - rec)        : {base_mean - rec_mean:+.4f}")
        print(f"  Wilcoxon p (paired)      : {p:.4f}")
        print(f"  rec better/worse/tie     : {better}/{worse}/{tie_n}\n")

if __name__ == "__main__":
    main()