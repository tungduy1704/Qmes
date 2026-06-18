"""scripts/reg/5_inference_reg.py

End-to-end inference cho regression:
  1. Load hold-out datasets (data/reg/inference.py)
  2. Oracle: true R² cho 7 circuit / dataset → pivot_r2_inference (cache)
  3. Load bundles via PairwiseRecommender.load() (4_select_save.py)
  4. Extract meta-features → recommend → so với true tied set
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from Qmes.config import TIED_THRESHOLD
from Qmes.data.reg.inference import load_inference_reg_datasets
from Qmes.evaluators.regression import RegressionEvaluator
from Qmes.extractors.regression import RegressionExtractor
from Qmes.recommender.pairwise import PairwiseRecommender
from Qmes.circuits.registry import get_circuit_names

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("inference_reg")

ROOT = Path(__file__).resolve().parents[2]
PIVOT_INF = ROOT / "results" / "pivot_r2_inference_600samples.csv"

BUNDLES = {
    "kNN_top10": ROOT / "artifacts_reg" / "kNN_top10",
    "NaiveBayes_full": ROOT / "artifacts_reg" / "NaiveBayes_full",
}

TOP_K = 3

def build_inference_pivot(datasets, circuits):
    if PIVOT_INF.exists():
        pivot = pd.read_csv(PIVOT_INF, index_col=0)
        if set(pivot.columns) == set(datasets) and set(pivot.index) == set(circuits):
            logger.info("Đọc lại pivot inference đã cache: %s", pivot.shape)
            return pivot
        logger.info("Cache không khớp dataset/circuit, tính lại")

    evaluator = RegressionEvaluator(n_splits=3, max_features=4, random_state=42)
    pivot = pd.DataFrame(index=circuits)
    pivot.index.name = "circuit"
    for name, (X, y) in datasets.items():
        logger.info("Oracle: %s %s", name, X.shape)
        scores = evaluator.evaluate_all(X, y, circuits)
        pivot[name] = [scores[c].get("mean_r2", np.nan) for c in circuits]
        pivot.to_csv(PIVOT_INF)
    return pivot

def _source(name: str) -> str:
    return "Synthetic" if any(k in name for k in ["quad", "sin", "friedman"]) else "OpenML"

def main():
    datasets = load_inference_reg_datasets()
    circuits = get_circuit_names()

    # ── 1. Ground truth ──────────────────────────────────────────
    pivot = build_inference_pivot(datasets, circuits)
    pivot = pivot.loc[circuits, list(datasets.keys())]

    tied_sets, true_best, max_r2 = {}, {}, {}
    for ds in datasets:
        col = pivot[ds]
        max_r2[ds] = col.max()
        tied_sets[ds] = set(col[col >= col.max() - TIED_THRESHOLD].index)
        true_best[ds] = col.idxmax()

    print("\n=== Ground truth (Oracle) ===")
    for ds in datasets:
        ceil = "CEILING" if len(tied_sets[ds]) == len(circuits) else ""
        print(f"  {ds:<22} best={true_best[ds]:<6} max={max_r2[ds]:+.3f} "
              f"tie={len(tied_sets[ds])} {ceil}")

    # ── 2. Extract meta-features ─────────────────────────────────
    extractor = RegressionExtractor()
    meta_vectors = {ds: extractor.extract(X, y) for ds, (X, y) in datasets.items()}

    # ── 3. Per bundle: recommend & evaluate ──────────────────────
    for bname, bdir in BUNDLES.items():
        if not (bdir / "recommender.pkl").exists():
            logger.warning("Bundle doesnot exist: %s — ignore", bdir)
            continue
        rec = PairwiseRecommender.load(bdir)

        if rec.task_type is not None and rec.task_type != extractor.task_type:
            raise ValueError(
                f"Task-type mismatch ({bname}):\n"
                f"  recommender: {rec.task_type}\n"
                f"  extractor  : {extractor.task_type}"
            )

        print(f"\n{'=' * 60}\n{bname}\n{'=' * 60}")
        rows = []
        for ds in datasets:
            res = meta_vectors[ds]

            full_order = rec.feature_names
            if full_order is not None and list(res.feature_names) != list(full_order):
                raise ValueError(
                    f"Feature-order mismatch ({ds}):\n"
                    f"  extractor : {res.feature_names}\n"
                    f"  bundle    : {full_order}"
                )
            pred = rec.predict(res.vector, top_k=TOP_K)
            top1 = pred["top_k"][0]
            top3 = pred["top_k"]

            rows.append({
                "dataset": ds,
                "source": _source(ds),
                "rec_top1": top1,
                "rec_top3": " > ".join(top3),
                "true_best": true_best[ds],
                "tied_hit": top1 in tied_sets[ds],
                "top3_hit": any(c in tied_sets[ds] for c in top3),
                "regret": max_r2[ds] - pivot.loc[top1, ds],
            })

        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

        n = len(df)
        print(f"\n  [ALL] n={n}  "
              f"Tied={df.tied_hit.mean():.4f}  "
              f"Top3={df.top3_hit.mean():.4f}  "
              f"Regret={df.regret.mean():.4f}")
        print("  by source:")
        print(df.groupby("source")[["tied_hit", "top3_hit", "regret"]]
              .mean().round(4).to_string())

if __name__ == "__main__":
    main()