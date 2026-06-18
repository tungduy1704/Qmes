"""scripts/clf/4_select_save.py

Retrain 2 selected configs on the filtered (non-degenerate) dataset,
save each model via PairwiseRecommender.save() into its own subdirectory.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from Qmes.config import TIED_THRESHOLD
from Qmes.evaluators.classification import filter_degenerate_datasets
from Qmes.recommender.selection import DEFAULT_CLASSIFIERS, select_features_mi
from Qmes.recommender.pairwise import PairwiseRecommender

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("select_and_save_clf")

ROOT = Path(__file__).resolve().parents[2]
META_PATH = ROOT / "results" / "meta_dataset_classification_single_avg_600samples.csv"
PIVOT_PATH = ROOT / "results" / "pivot_mcc_classification_600samples.csv"
SUMMARY_PATH = ROOT / "results" / "recommender_clf_summary_600samples.csv" 
ART_DIR = ROOT / "artifacts_clf"

TASK_TYPE = "classification"
METRIC_NAME = "MCC"
K_VALUES = [5, 10, 15, 20]

# (config_name, classifier_key, feature_subset_label)
CONFIGS = [
    ("kNN_top5", "kNN",  "top5"),
    ("kNN_top10", "kNN",  "top10"),
]

def get_loo_metrics(summary: pd.DataFrame, clf_key: str, feat_label: str) -> dict:
    row = summary[(summary["Features"] == feat_label) & (summary["Classifier"] == clf_key)]
    if row.empty:
        logger.warning("Cannot find LOO metrics for %s/%s in summary", clf_key, feat_label)
        return {}
    r = row.iloc[0]
    return {
        "Tied": float(r["Tied"]),
        "Top3_Tied": float(r["Top3_Tied"]),
        "Mean_Regret": float(r["Mean_Regret"]),
    }


def main():
    meta = pd.read_csv(META_PATH, index_col=0)
    pivot = pd.read_csv(PIVOT_PATH, index_col=0)
    summary = pd.read_csv(SUMMARY_PATH)

    common = [ds for ds in pivot.columns if ds in meta.index]
    assert len(common) == len(pivot.columns) == len(meta.index), "align mismatch"

    pivot_f, reasons = filter_degenerate_datasets(
        pivot, min_max_score=0.1, ceiling_threshold=0.99
    )
    datasets = list(pivot_f.columns)
    meta_f = meta.loc[pivot_f.columns]
    X_all = meta_f.values
    feature_names_all = list(meta_f.columns)   

    print(f"Filtered {len(reasons)}, training on {len(datasets)} datasets")
    print(f"Feature order: {feature_names_all}\n")

    # MI subsets — same seed and labels as LOO to ensure consistency
    y_single = pivot_f.idxmax(axis=0).values
    subsets = select_features_mi(X_all, y_single, k_values=K_VALUES)

    ART_DIR.mkdir(parents=True, exist_ok=True)

    for cfg_name, clf_key, feat_label in CONFIGS:
        feat_idx = subsets[feat_label]
        feat_names = [feature_names_all[i] for i in feat_idx]   # subset thực sự dùng
        loo_metrics = get_loo_metrics(summary, clf_key, feat_label)

        print(f"=== {cfg_name} ({clf_key}/{feat_label}, {len(feat_idx)} features) ===")
        print(f"  features: {feat_names}")
        print(f"  loo_metrics: {loo_metrics}")

        rec = PairwiseRecommender(
            classifier=DEFAULT_CLASSIFIERS[clf_key],
            feature_indices=feat_idx,
            tied_threshold=TIED_THRESHOLD,
            feature_names=feature_names_all,
            task_type=TASK_TYPE,
            metric_name=METRIC_NAME,
        )
        rec.fit(X_all, pivot_f)

        sample_pred = rec.predict(X_all[0], top_k=3)
        tied0 = set(
            pivot_f[datasets[0]][pivot_f[datasets[0]] >= pivot_f[datasets[0]].max() - TIED_THRESHOLD].index
        )
        print(f"  sample[{datasets[0]}] top3: {sample_pred['top_k']}, true tied: {tied0}")

        out_dir = ART_DIR / cfg_name
        rec.save(out_dir)

        config_meta = {
            "config_name": cfg_name,
            "classifier": clf_key,
            "feature_subset": feat_label,
            "tied_threshold": TIED_THRESHOLD,
            "n_train_datasets": len(datasets),
            "selected_feature_names": feat_names,
            "loo_metrics": loo_metrics,
        }
        with open(out_dir / "config_meta.json", "w", encoding="utf-8") as f:
            json.dump(config_meta, f, indent=2, ensure_ascii=False)
        print(f"  saved → {out_dir}\n")


if __name__ == "__main__":
    main()