"""Oracle classification: 108 datasets × 7 circuits → results/pivot_mcc_classification.csv

- Incremental save: lưu sau mỗi dataset, crash giữa chừng không mất công, chạy lại tự resume.
- Checkpoint: so MCC với backup trên nhóm dataset KHÔNG bị cap (data không đổi → phải trùng khớp).
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from Qmes.data.classification import load_classification_datasets
from Qmes.evaluators.classification import ClassificationEvaluator
from Qmes.circuits.registry import get_circuit_names

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("run_oracle_clf")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pivot_mcc_classification.csv"
BACKUP = ROOT / "backup" / "pivot_mcc_classification_clean.csv"


def main():
    datasets = load_classification_datasets()
    evaluator = ClassificationEvaluator(n_splits=3, max_features=4, random_state=42)
    circuits = get_circuit_names()

    # Resume: bỏ qua dataset đã có trong file output
    if OUT.exists():
        pivot = pd.read_csv(OUT, index_col=0)
        done = set(pivot.columns)
        logger.info("Resume: %d datasets đã xong", len(done))
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
        pivot.to_csv(OUT)  # lưu sau MỖI dataset

    # ── Checkpoint: so với backup trên nhóm KHÔNG bị cap ──
    print("\n=== CHECKPOINT ===")
    print(f"Pivot shape: {pivot.shape}  (kỳ vọng (7, 108))")
    nan_count = int(pivot.isna().sum().sum())
    print(f"NaN entries: {nan_count}")

    if BACKUP.exists():
        old = pd.read_csv(BACKUP, index_col=0)
        uncapped = [n for n, (X, _) in datasets.items() if len(X) < 300]
        common = [n for n in uncapped if n in old.columns and n in pivot.columns]
        if common:
            diff = (pivot[common] - old.loc[pivot.index, common]).abs()
            print(f"So sánh {len(common)} dataset không bị cap với backup:")
            print(f"  max |ΔMCC| = {diff.max().max():.6f}  (kỳ vọng ≈ 0)")
            bad = diff.columns[(diff > 6e-5).any()].tolist()
            if bad:
                print(f"  ⚠ Lệch bất thường ở: {bad}")
        else:
            print("(Không có dataset chung để so — kiểm tra tên cột trong backup)")
    else:
        print("(Không tìm thấy backup để so sánh)")


if __name__ == "__main__":
    main()