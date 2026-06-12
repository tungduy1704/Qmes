import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
new = pd.read_csv(ROOT / "results" / "pivot_mcc_classification.csv", index_col=0)
old = pd.read_csv(ROOT / "backup" / "pivot_mcc_classification_clean.csv", index_col=0)

# 8 dataset KHÔNG bị cap trong 20 cái đã xong
uncapped = ["Iris_01", "Iris_02", "Iris_12", "Wine_01", "Wine_02", "Wine_12",
            "Statlog Heart", "Spect Heart"]
common = [c for c in uncapped if c in new.columns and c in old.columns]
print(f"So sánh được {len(common)}/{len(uncapped)}: {common}")

if common:
    diff = (new[common] - old.loc[new.index, common]).abs()
    print(diff.round(6))
    print(f"\nmax |ΔMCC| = {diff.max().max():.6f}  (kỳ vọng ≈ 0)")