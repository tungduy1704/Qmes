"""scripts/clf/5_inference.py

End-to-end inference for classification:
  1. Load hold-out datasets (data/clf/inference.py)
  2. Oracle: true MCC for 7 circuits per dataset → pivot_mcc_inference.csv (cache)
  3. Load 2 bundles via PairwiseRecommender.load() (4_select_save.py)
  4. Extract meta-features → recommend → compare against true tied set
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from Qmes.data.clf.inference import load_inference_classification
from Qmes.evaluators.classification import ClassificationEvaluator
from Qmes.extractors.classification import ClassificationExtractor
from Qmes.recommender.pairwise import PairwiseRecommender
from Qmes.circuits.registry import get_circuit_names

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("inference_clf")

ROOT = Path(__file__).resolve().parents[2]
ART_DIR = ROOT / "results"
PIVOT_INF = ROOT / "results" / "pivot_mcc_inference_600samples.csv"
BUNDLES = {
    "kNN_top5": ROOT / "artifacts_clf" / "kNN_top5",
    "kNN_top10": ROOT / "artifacts_clf" / "kNN_top10",
}

TIED_THRESHOLD = 0.01
TOP_K = 3

def build_inference_pivot(datasets, circuits):
    if PIVOT_INF.exists():
        pivot = pd.read_csv(PIVOT_INF, index_col=0)
        if set(pivot.columns) == set(datasets) and set(pivot.index) == set(circuits):
            logger.info("Đọc lại pivot inference đã cache: %s", pivot.shape)
            return pivot
        logger.info("Cache không khớp dataset/circuit, tính lại")

    evaluator = ClassificationEvaluator(n_splits=3, max_features=4, random_state=42)
    pivot = pd.DataFrame(index=circuits)
    pivot.index.name = "circuit"
    for name, (X, y) in datasets.items():
        logger.info("Oracle: %s %s", name, X.shape)
        scores = evaluator.evaluate_all(X, y, circuits)
        pivot[name] = [scores[c].get("mean_mcc", np.nan) for c in circuits]
        pivot.to_csv(PIVOT_INF)  
    return pivot

def main():
    datasets, signal_flags = load_inference_classification()
    circuits = get_circuit_names()

    # ── 1. Ground truth ──────────────────────────────────────────
    pivot = build_inference_pivot(datasets, circuits)
    pivot = pivot.loc[circuits, list(datasets.keys())]  

    tied_sets, true_best, max_mcc = {}, {}, {}
    for ds in datasets:
        col = pivot[ds]
        max_mcc[ds] = col.max()
        tied_sets[ds] = set(col[col >= col.max() - TIED_THRESHOLD].index)
        true_best[ds] = col.idxmax()

    print("\n=== Ground truth (Oracle) ===")
    for ds in datasets:
        ceil = "CEILING" if len(tied_sets[ds]) == len(circuits) else ""
        flag = "" if signal_flags[ds] else "[ceil-flag]"
        print(f"  {ds:<20} best={true_best[ds]:<6} max={max_mcc[ds]:.3f} "
              f"tie={len(tied_sets[ds])} {flag}{ceil}")

    # ── 2. Extract meta-features ──
    extractor = ClassificationExtractor()
    meta_vectors = {}
    for ds, (X, y) in datasets.items():
        meta_vectors[ds] = extractor.extract(X, y)  

    # 3. Per bundle: recommend and evaluate against true tied set
    for bname, bdir in BUNDLES.items():
        rec = PairwiseRecommender.load(bdir)

        if rec.task_type is not None and rec.task_type != extractor.task_type:
            raise ValueError(
                f"Task-type mismatch ({bname}):\n"
                f"  recommender: {rec.task_type}\n"
                f"  extractor  : {extractor.task_type}"
            )

        print(f"\n{'='*60}\n{bname}\n{'='*60}")
        rows = []
        for ds in datasets:
            res = meta_vectors[ds]

            full_order = rec.feature_names
            if full_order is not None and list(res.feature_names) != list(full_order):
                raise ValueError(
                    f"Feature-order mismatch ({ds}):\n"
                    f"  extractor: {res.feature_names}\n"
                    f"  bundle   : {full_order}"
                )
            pred = rec.predict(res.vector, top_k=TOP_K)
            top1 = pred["top_k"][0]
            top3 = pred["top_k"]

            rows.append({
                "dataset": ds,
                "signal": signal_flags[ds],
                "rec_top1": top1,
                "rec_top3": " > ".join(top3),
                "true_best": true_best[ds],
                "tied_hit": top1 in tied_sets[ds],
                "top3_hit": any(c in tied_sets[ds] for c in top3),
                "regret": max_mcc[ds] - pivot.loc[top1, ds],
            })

        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

        for label, sub in [("SIGNAL only", df[df.signal]),
                            ("ALL (incl ceiling)", df)]:
            n = len(sub)
            print(f"\n  [{label}] n={n}  "
                  f"Tied={sub.tied_hit.mean():.4f}  "
                  f"Top3={sub.top3_hit.mean():.4f}  "
                  f"Regret={sub.regret.mean():.4f}")

if __name__ == "__main__":
    main()