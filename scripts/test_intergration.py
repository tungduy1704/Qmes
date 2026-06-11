"""Integration test: verify full pipeline on Iris binary."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

print("=" * 60)
print("INTEGRATION TEST — Qmes pipeline")
print("=" * 60)

# ── 1. Prepare small dataset ─────────────────────────────────
from Qmes.data.preprocessing import encode_categoricals, impute_and_cast

iris = load_iris()
mask = iris.target < 2  # binary: class 0 vs 1
X = iris.data[mask][:80]  # small subset
y = iris.target[mask][:80]
print(f"\n[1] Dataset: Iris 0v1, shape={X.shape}")

# ── 2. Extractor ─────────────────────────────────────────────
from Qmes.extractors import get_extractor

extractor = get_extractor("classification")
result = extractor.extract(X, y)
print(f"\n[2] Extractor OK")
print(f"    task_type = {result.task_type}")
print(f"    vector dim = {result.dim}")
print(f"    features = {result.feature_names[:5]}...")
assert result.dim == len(result.feature_names)
assert np.all(np.isfinite(result.vector))
print("    PASS")

# ── 3. Evaluator (2 circuits only, keep fast) ────────────────
from Qmes.evaluators import ClassificationEvaluator

evaluator = ClassificationEvaluator(n_splits=2, max_features=4)
test_circuits = ["unit", "SRx"]

print(f"\n[3] Evaluator — testing {test_circuits}")
for c_name in test_circuits:
    scores = evaluator.evaluate_circuit(X, y, c_name)
    print(f"    {c_name}: MCC={scores['mean_mcc']:.4f}, "
          f"Acc={scores['mean_acc']:.4f}")
    assert -1 <= scores["mean_mcc"] <= 1
    assert 0 <= scores["mean_acc"] <= 1
print("    PASS")

# ── 4. Evaluator — evaluate_all + build_pivot ────────────────
print(f"\n[4] evaluate_all on 2 circuits")
all_scores = evaluator.evaluate_all(X, y, circuit_names=test_circuits)
print(f"    Results: {list(all_scores.keys())}")
assert len(all_scores) == 2
print("    PASS")

# ── 5. Recommender fit/predict (mock pivot) ──────────────────
from Qmes.circuits import get_circuit_names
from Qmes.recommender import PairwiseRecommender
from sklearn.ensemble import GradientBoostingClassifier

print(f"\n[5] Recommender fit/predict")

# Fake pivot: 7 circuits × 3 datasets, random scores
circuits = get_circuit_names()
# Thay đoạn fake pivot trong test
fake_datasets = [f"ds_{i}" for i in range(15)]
rng = np.random.RandomState(42)
pivot_fake = pd.DataFrame(
    rng.rand(len(circuits), 15),
    index=circuits,
    columns=fake_datasets,
)

d = result.dim
meta_fake = pd.DataFrame(
    rng.rand(15, d),
    index=fake_datasets,
    columns=result.feature_names,
)

rec = PairwiseRecommender(
    classifier=GradientBoostingClassifier(random_state=42),
    feature_indices=None,
)
rec.fit(meta_fake, pivot_fake)
print(f"    Fitted: {len(rec.classifiers_)} pairwise classifiers")

pred = rec.predict(result.vector, top_k=3)
print(f"    Prediction: top3={pred['top_k']}, votes={pred['votes']}")
assert len(pred["top_k"]) == 3
assert all(c in circuits for c in pred["top_k"])
print("    PASS")

# ── 6. Recommender save/load ─────────────────────────────────
from pathlib import Path
import shutil

print(f"\n[6] Save/Load")
tmp_path = Path("_test_artifacts")
rec.save(tmp_path)
rec2 = PairwiseRecommender.load(tmp_path)
pred2 = rec2.predict(result.vector, top_k=3)
assert pred2["top_k"] == pred["top_k"]
shutil.rmtree(tmp_path)
print("    Save/Load roundtrip PASS")

# ── 7. Inference recommend() ─────────────────────────────────
from Qmes.inference import recommend

print(f"\n[7] Inference recommend()")
rec_result = recommend(
    X, y,
    extractor=extractor,
    recommender=rec,
    top_k=3,
    preprocess=False,  # already clean
)
print(f"    Result: {rec_result['top_k']}")
assert "top_k" in rec_result
assert "votes" in rec_result
assert "meta_features" in rec_result
print("    PASS")

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)