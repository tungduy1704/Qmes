import pandas as pd
from pathlib import Path
from Qmes.data.classification import load_classification_datasets

ROOT = Path(__file__).resolve().parents[1]
new = pd.read_csv(ROOT / "results" / "pivot_mcc_classification.csv", index_col=0)
old = pd.read_csv(ROOT / "backup" / "pivot_mcc_classification_clean.csv", index_col=0)

datasets = load_classification_datasets()
capped = [n for n, (X, _) in datasets.items()
          if len(X) == 300 and n in old.columns and n in new.columns]

delta = (new[capped] - old.loc[new.index, capped])
print(f"=== {len(capped)} dataset bị cap ===")
print(f"Mean ΔMCC (mới - cũ): {delta.mean().mean():.4f}")   # kỳ vọng ÂM (hết thổi phồng)
print(f"Min / Max ΔMCC:       {delta.min().min():.4f} / {delta.max().max():.4f}")

changed = [(ds, old[ds].idxmax(), new[ds].idxmax())
           for ds in capped if old[ds].idxmax() != new[ds].idxmax()]
print(f"\nNhãn best-circuit đổi: {len(changed)}/{len(capped)} dataset")
for ds, o, n in changed:
    print(f"  {ds:25s} {o:8s} → {n}")