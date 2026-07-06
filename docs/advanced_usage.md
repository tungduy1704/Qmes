# Advanced Usage

This page covers building your own extractor, circuit, and recommender -
the pieces you'd use to extend Qmes beyond the bundled 7 circuits and
22/12-dim Problexity meta-features. Every code block on this page has been
run end-to-end; outputs shown are the actual printed results.

## 1. Custom extractor

`BaseExtractor` is the abstract base every meta-feature extractor implements.
The contract is small - three members:

- `task_type` (property): a string identifier (`"classification"`, `"regression"`, ...)
- `_feature_names` (property): fixed-length list of feature names - its length defines the output dimension
- `_extract_raw(X, y)`: compute the raw meta-feature vector

The public `extract(X, y)` method wraps `_extract_raw`: it validates `X` is
2D with at least 2 samples, pads/truncates the output to match
`_feature_names`, replaces any NaN/Inf with `0.0`, and returns an
`ExtractionResult(vector, feature_names, task_type)`.

```python
import numpy as np
from Qmes.extractors.base import BaseExtractor

class TinyExtractor(BaseExtractor):
    @property
    def task_type(self):
        return "classification"

    @property
    def _feature_names(self):
        return ["n_samples", "n_features", "class_balance"]

    def _extract_raw(self, X, y=None):
        if y is None:
            raise ValueError("need y")
        n_samples, n_features = X.shape
        classes, counts = np.unique(y, return_counts=True)
        balance = counts.min() / counts.max()
        return np.array([n_samples, n_features, balance], dtype=np.float64)

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=50, n_features=5, random_state=0)
ext = TinyExtractor()
res = ext.extract(X, y)
print(res.vector, res.feature_names, res.task_type, res.dim)
# [50.  5.  1.] ['n_samples', 'n_features', 'class_balance'] classification 3
print(res.to_dict())
# {'n_samples': 50.0, 'n_features': 5.0, 'class_balance': 1.0}
```

### Where `feature_names` matters downstream

`PairwiseRecommender` stores the `feature_names` it was trained with and
`recommend()` asserts the extractor's `result.feature_names` matches it
**exactly, positionally** - this is the alignment guard that survives
`save()`/`load()`. Fit a tiny recommender on this extractor's 3 features and
run it through `recommend()`:

```python
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from Qmes import get_recommender, recommend

# Build 3 fake datasets and extract meta-features for each
names, rows = [], []
for i, (ns, w) in enumerate([(50, 1.0), (200, 0.5), (80, 0.8)]):
    Xd, yd = make_classification(
        n_samples=ns, n_features=5,
        weights=[w, 1 - w] if w < 1 else None, random_state=i,
    )
    names.append(f"ds{i}")
    rows.append(ext.extract(Xd, yd).vector)
meta = np.vstack(rows)

circuits = ["unit", "RY", "ZFM"]
pivot = pd.DataFrame(
    np.random.RandomState(0).rand(len(circuits), len(names)),
    index=circuits, columns=names,
)

rec = get_recommender(
    "classification", KNeighborsClassifier(n_neighbors=1),
    feature_names=ext._feature_names,
)
rec.fit(meta, pivot)

X_new, y_new = make_classification(n_samples=120, n_features=5, random_state=99)
out = recommend(X_new, y_new, extractor=ext, recommender=rec)
print(out["ranking"], out["top_k"], out["votes"])
# ['ZFM', 'RY', 'unit'] ['ZFM', 'RY', 'unit'] {'unit': 0, 'RY': 1, 'ZFM': 2}
```

If the recommender's `feature_names` doesn't match the extractor's - same
length, different order, whatever - `recommend()` raises before anything
downstream runs silently on misaligned columns:

```python
rec_bad = get_recommender(
    "classification", KNeighborsClassifier(n_neighbors=1),
    feature_names=["n_features", "n_samples", "class_balance"],  # swapped!
)
rec_bad.fit(meta, pivot)
recommend(X_new, y_new, extractor=ext, recommender=rec_bad)
# ValueError: Feature-name mismatch between extractor and recommender:
#   extractor : ['n_samples', 'n_features', 'class_balance']
#   recommender: ['n_features', 'n_samples', 'class_balance']
```

## 2. Custom circuit + evaluator

Circuits live in `Qmes.circuits.registry.CIRCUIT_POOL`, a plain
`{name: callable}` dict. Registering a new one means adding an entry - no
subclassing required. This example wires in `HZY_CZ_encode`, one of the
parameterized encoders already in `Qsun.Qencodes`, with frozen random
parameters so the circuit is deterministic:

```python
import Qmes.circuits.registry as registry
from Qmes.evaluators.classification import ClassificationEvaluator
from Qmes.Qsun.Qencodes import HZY_CZ_encode

rng = np.random.RandomState(7)
fixed_params = rng.rand(20)

def hzy_encode(x):
    return HZY_CZ_encode(x, params=fixed_params, n_layers=2)

registry.CIRCUIT_POOL["HZY"] = hzy_encode
# Correct registration: HZY_CZ_encode rotates by the raw feature value in
# radians, so it expects the default (0, pi) range -- "HZY" must NOT be
# added to UNIT_RANGE_CIRCUITS.

X, y = make_classification(
    n_samples=150, n_features=4, n_informative=3, n_redundant=0,
    n_clusters_per_class=1, class_sep=1.5, random_state=3,
)
evaluator = ClassificationEvaluator(n_splits=5)
scores = evaluator.evaluate_circuit(X, y, "HZY")
print({k: round(v, 4) for k, v in scores.items()})
# {'mean_acc': 0.9933, 'std_acc': 0.0133, 'mean_mcc': 0.9871, 'std_mcc': 0.0258,
#  'mean_f1': 0.9935, 'std_f1': 0.0129}
```

!!! warning "The UNIT_RANGE_CIRCUITS trap"
    `ClassificationEvaluator`/`RegressionEvaluator` pick the `MinMaxScaler`
    range per circuit: `(0, 1)` if the circuit's name is in
    `UNIT_RANGE_CIRCUITS`, otherwise `(0, π)`. Only `"unit"` needs `(0, 1)`
    out of the box - it computes `sqrt(x)` and `sqrt(1-x)`, which is
    undefined outside `[0, 1]`.

    **If you register a new circuit and get its range wrong, nothing
    raises.** The pipeline runs to completion and returns a number that
    looks like a real score. The only symptom is a kernel that has
    quietly lost its ability to tell datapoints apart.

    Here's the actual effect on `HZY`, measured at the point where the
    evaluator computes the kernel matrix — same data, same circuit, same
    frozen params, only the `MinMaxScaler` range differs:

    ```python
    from Qmes.circuits.registry import compute_kernel_matrix
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    Xs = StandardScaler().fit_transform(X)
    X_correct = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(Xs)
    X_mistake = MinMaxScaler(feature_range=(0, 1)).fit_transform(Xs)  # forgot HZY needs (0, pi)

    K_correct = compute_kernel_matrix(X_correct, X_correct, hzy_encode)
    K_mistake = compute_kernel_matrix(X_mistake, X_mistake, hzy_encode)

    iu = np.triu_indices(len(X), 1)
    print(f"correct : mean={K_correct[iu].mean():.4f} std={K_correct[iu].std():.4f}")
    # correct : mean=0.1610 std=0.2277
    print(f"mistake : mean={K_mistake[iu].mean():.4f} std={K_mistake[iu].std():.4f}")
    # mistake : mean=0.7618 std=0.1456
    ```

    Under the wrong range, every pair of points looks 0.76-similar on
    average with much less spread (std 0.15 vs 0.23) - the kernel has
    collapsed toward "everything looks the same," which starves the SVC of
    signal. No exception, no warning: just a quietly worse recommender if
    this circuit's scores end up in a training pivot.

    **Rule of thumb:** if your circuit maps feature values directly into
    rotation angles (most of them do), it wants `(0, π)` and needs no
    entry in `UNIT_RANGE_CIRCUITS`. Only add a circuit there if it
    specifically expects inputs already in `[0, 1]`, the way `unit_encode`'s
    amplitude encoding does.

## 3. Retrain a `PairwiseRecommender`

The full offline loop: extract meta-features for a batch of datasets, build
a ground-truth pivot with an `Evaluator`, fit a `PairwiseRecommender`, and
persist it.

```python
from Qmes import get_extractor, get_evaluator, get_recommender, recommend

CIRCUITS = ["unit", "RY", "ZFM"]

raw_datasets = {
    "ds0": make_classification(n_samples=80, n_features=4, n_informative=3, n_redundant=0, random_state=0),
    "ds1": make_classification(n_samples=80, n_features=4, n_informative=2, n_redundant=1, random_state=1, class_sep=0.7),
    "ds2": make_classification(n_samples=80, n_features=4, n_informative=3, n_redundant=0, random_state=2, weights=[0.3, 0.7]),
    "ds3": make_classification(n_samples=80, n_features=4, n_informative=2, n_redundant=1, random_state=3, class_sep=1.8),
    "ds4": make_classification(n_samples=80, n_features=4, n_informative=3, n_redundant=0, random_state=4, flip_y=0.1),
}

extractor = get_extractor("classification")
evaluator = get_evaluator("classification", n_splits=3)

meta_df = extractor.extract_batch(raw_datasets)                      # index=dataset names
pivot = evaluator.build_pivot(raw_datasets, circuit_names=CIRCUITS)   # index=circuits, columns=datasets

# fit() aligns meta rows to pivot columns POSITIONALLY, not by name/index
meta_aligned = meta_df.loc[pivot.columns]

rec = get_recommender(
    "classification", KNeighborsClassifier(n_neighbors=1),
    feature_names=extractor._feature_names,
)
rec.fit(meta_aligned.values, pivot)
rec.save("my_bundle")
```

`save()` writes `recommender.npz` (the training meta-feature matrix and
pivot values) and `recommender.json` (the classifier's class + `get_params()`,
plus `feature_names`/`task_type`/etc.). It does **not** pickle the fitted
estimator. `load()` reconstructs the classifier from the spec and **refits**
it on the stored training data:

```python
rec_1 = type(rec).load("my_bundle")
rec_2 = type(rec).load("my_bundle")

X_new, y_new = make_classification(n_samples=100, n_features=4, n_informative=3, n_redundant=0, random_state=99)
out1 = recommend(X_new, y_new, extractor=extractor, recommender=rec_1)
out2 = recommend(X_new, y_new, extractor=extractor, recommender=rec_2)
print(out1["ranking"] == out2["ranking"])  # True
```

### Gotcha: non-JSON-serializable classifier params

`save()` calls `json.dump()` on `classifier.get_params()`. If any param is
itself an object — e.g. `AdaBoostClassifier(estimator=DecisionTreeClassifier())` -
this fails immediately with a `TypeError`, not a confusing error somewhere
in `load()` later:

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

bad_rec = get_recommender(
    "classification",
    AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=42), random_state=42),
    feature_names=extractor._feature_names,
)
bad_rec.fit(meta_aligned.values, pivot)
bad_rec.save("bad_bundle")
# TypeError: Object of type DecisionTreeClassifier is not JSON serializable
```

### Gotcha: refit-on-load is only deterministic if the classifier is

`load()`'s refit is cheap and exactly reproducible for the bundled kNN
because kNN has no internal randomness. That is **not** true in general.
If you retrain with a stochastic classifier like `RandomForestClassifier`
and don't fix `random_state`, each `load()` call reseeds independently and
refits to a genuinely different model - same training data, same code,
different result:

```python
from sklearn.ensemble import RandomForestClassifier

query_vector = extractor.extract(X_new, y_new).vector

rec_rf = get_recommender(
    "classification", RandomForestClassifier(n_estimators=10),  # random_state=None
    feature_names=extractor._feature_names,
)
rec_rf.fit(meta_aligned.values, pivot)
rec_rf.save("rf_bundle_unseeded")

for _ in range(5):
    loaded = type(rec_rf).load("rf_bundle_unseeded")
    print(loaded.classifiers_[("unit", "RY")].predict_proba(np.atleast_2d(query_vector)))
# [[0.5 0.5]]
# [[0.9 0.1]]
# [[0.9 0.1]]
# [[0.9 0.1]]
# [[0.6 0.4]]
```

`random_state=None` is itself part of `get_params()`, so it's faithfully
persisted and passed back to the constructor on load - but `None` means
"reseed randomly," so persisting it buys you nothing. Fix `random_state`
and the same loop is exactly reproducible:

```python
rec_rf_seeded = get_recommender(
    "classification", RandomForestClassifier(n_estimators=10, random_state=42),
    feature_names=extractor._feature_names,
)
rec_rf_seeded.fit(meta_aligned.values, pivot)
rec_rf_seeded.save("rf_bundle_seeded")

for _ in range(5):
    loaded = type(rec_rf_seeded).load("rf_bundle_seeded")
    print(loaded.classifiers_[("unit", "RY")].predict_proba(np.atleast_2d(query_vector)))
# [[0.9 0.1]]
# [[0.9 0.1]]
# [[0.9 0.1]]
# [[0.9 0.1]]
# [[0.9 0.1]]
```

**Takeaway:** a saved bundle reproduces the exact same recommender on load
if and only if its classifier is deterministic (like kNN) or has a fixed
`random_state`. If you retrain with a stochastic classifier and skip
`random_state`, treat every `load()` as a fresh (similar, not identical)
model.
