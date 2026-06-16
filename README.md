# Qmes

**Q**uantum **M**eta-learning for **E**ncoding **S**election — a meta-learning framework that recommends a suitable quantum encoding circuit for a given dataset **without running any quantum evaluation at inference time**.

Given a new dataset `(X, y)` and a task type, Qmes extracts classical complexity meta-features and uses an offline-trained classical meta-learner to predict which encoding circuit (from a fixed pool) is most likely to perform best — collapsing what would otherwise be an expensive per-circuit quantum search into a single forward pass.

> **Status:** research code (v0.1.0). Classification and Regression pipelines are implemented end-to-end. Other task types are planned (see [Roadmap](#roadmap)).

---

## Idea in one paragraph

Choosing a quantum feature map for a dataset normally means evaluating every candidate circuit on that dataset — expensive, and it has to be redone for each new dataset. Qmes treats this as an **algorithm-selection problem** (in the sense of Rice, 1976): build an offline meta-dataset by evaluating every circuit on many benchmark datasets, describe each dataset with cheap classical complexity metrics, and train a classical model to map *meta-features → suitable circuit*. At inference time only the cheap classical part runs.

---

## Architecture

Three pluggable components, each an abstract base class with one concrete implementation per task type:

| Component | Base class | Role |
|---|---|---|
| **Extractor** | `BaseExtractor` | Compute a fixed-length meta-feature vector from a dataset |
| **Evaluator** (Oracle) | `BaseEvaluator` | Score every circuit on a dataset to produce meta-labels (offline only) |
| **Recommender** | `PairwiseRecommender` | Classical meta-learner: predict a circuit ranking from meta-features |

```
                       OFFLINE (build meta-dataset + train)
   benchmark datasets ──► Extractor ──► meta-features ─┐
                      └─► Evaluator ──► circuit scores ─┴─► Recommender.fit()
                                                                 │
                                                                 ▼
                       INFERENCE (no quantum evaluation)   recommender.pkl
   new (X, y) ──► Extractor ──► meta-features ──► Recommender.predict() ──► ranked circuits
```

The split matters: the **Evaluator runs quantum kernel evaluations and is only used offline**. Inference touches the Extractor and Recommender only.

---

## Circuit pool

Seven encoding circuits, backed by the bundled **Qsun** simulator (`Qmes/Qsun/`):

| Name | Encoding |
|---|---|
| `unit` | Unit / amplitude-style encoding |
| `SRx` | Separable RX |
| `RY` | Angle encoding (RY) |
| `HERx` | Hardware-efficient embedding (RX) |
| `RY_CX` | RY with linear CX entanglement |
| `ZFM` | Z feature map |
| `HD` | High-dimensional encoding |

All circuits except `unit` scale inputs to `[0, π]`. The pool is defined in `Qmes/circuits/registry.py` and is intentionally easy to extend.

---

## Implemented task types

### Tabular classification
- **Extractor** (`ClassificationExtractor`): 22 Problexity complexity measures (Lorena et al.). Stochastic measures (`l3`, `n4`) are averaged over multiple seeds for determinism; data is min-max scaled internally as Problexity requires.
- **Evaluator** (`ClassificationEvaluator`): SVC with a **precomputed quantum fidelity kernel**, 3-fold stratified CV. Preprocessing per fold: `StandardScaler → PCA → MinMaxScaler` (PCA fit on the train split only). Primary metric: **MCC** (also reports accuracy, F1).
- **Recommender**: pairwise One-vs-One over the 7 circuits.

### Tabular regression
- **Extractor** (`RegressionExtractor`): 12 Problexity regression measures — `c1–c4` (correlation), `l1–l3` (linearity), `s1–s4` (smoothness), `t2` (dimensionality).
- **Evaluator** (`RegressionEvaluator`): KernelRidge with a **precomputed quantum fidelity kernel**, 3-fold KFold CV, same `StandardScaler → PCA → MinMaxScaler` order. Primary metric: **R²** (also reports RMSE, MAE).
- **Recommender**: pairwise One-vs-One.

> Note: both task types use **quantum-kernel methods** as the Oracle, not variational circuits. A fixed-kernel Oracle is deterministic and isolates the contribution of the *encoding* (no trainable ansatz parameters to confound the signal).

---

## The recommender (pairwise One-vs-One)

`PairwiseRecommender` trains one binary classifier for each of the C(7,2) = 21 circuit pairs. Each classifier learns, from meta-features, which of two circuits scores higher. At prediction time the 21 votes are aggregated into a full ranking.

- **Tied threshold** (`0.01`): two circuits are treated as tied if their metric values are within this absolute delta — used so the Oracle's "best" set isn't artificially narrow.
- **Model selection** (`recommender/selection.py`): exhaustive **Leave-One-Out** over (classifier × MI-selected feature subset). LOO accuracy on the meta-dataset is the validation signal — pairwise classifiers reaching training accuracy 1.0 is expected and not a sign of overfitting.

---

## Installation

```bash
git clone https://github.com/tungduy1704/Qmes.git
cd Qmes
pip install -e .
```

Requires **Python ≥ 3.10**.

> **Dependencies are not yet declared in `pyproject.toml`.** Until that's fixed, install the runtime deps manually:
> ```bash
> pip install numpy pandas scikit-learn problexity ucimlrepo
> ```
> Qsun is bundled inside the package — no separate install needed.

---

## Quick start (inference)

Plan to update soon!!!

---

## Offline pipeline (build meta-dataset + train)

Numbered scripts under `scripts/clf/` and `scripts/reg/` run the full offline workflow. Run them in order:

| Step | Script | Output |
|---|---|---|
| 1 | `1_extract.py` | Meta-features for all benchmark datasets |
| 2 | `2_evaluate.py` | Quantum-kernel circuit scores (the Oracle / meta-labels) |
| 3 | `3_train.py` | LOO model selection over classifiers × feature subsets |
| 4 | `4_select_save.py` | Fit and serialize the chosen recommender bundle(s) |
| 5 | `5_inference.py` | End-to-end evaluation on held-out datasets |
| 6 | `6_baseline.py` | Baseline comparisons |

---

## Repository layout

```
Qmes/                          # installable package
├── __init__.py                # sets __version__ only (no re-exports)
├── Qsun/                      # bundled quantum simulator
│   ├── Qencodes.py            # encoding circuits used by the pool
│   ├── Qkernels.py            # state_product / fidelity kernel
│   ├── Qcircuit.py  Qgates.py  Qmeas.py  Qwave.py  Qdata.py
│   ├── Qiqp.py                # IQP circuit class — bundled, NOT in the 7-circuit pool
│   └── qcit_gate_walk.py      # quantum-walk utility — legacy, unused by the pipeline
├── circuits/
│   └── registry.py            # CIRCUIT_POOL (7 circuits), compute_kernel_matrix()
├── data/
│   ├── preprocessing.py       # encode_categoricals, impute_and_cast, scale_features
│   ├── clf/                   # classification loaders: train.py + inference.py
│   └── reg/                   # regression loaders:     train.py + inference.py
├── extractors/
│   ├── base.py                # BaseExtractor (ABC)
│   ├── classification.py      # Problexity — 22 meta-features
│   └── regression.py          # Problexity — 12 meta-features
├── evaluators/
│   ├── base.py                # BaseEvaluator (ABC) — evaluate_circuit, evaluate_all, build_pivot
│   ├── classification.py      # SVC + quantum fidelity kernel, metric: MCC
│   └── regression.py          # KernelRidge + quantum fidelity kernel, metric: R²
├── recommender/
│   ├── pairwise.py            # PairwiseRecommender — fit, predict, save, load
│   └── selection.py           # DEFAULT_CLASSIFIERS, select_features_mi, run_loo_evaluation
└── inference/
    └── runner.py              # preprocess_new_dataset, recommend, evaluate_recommendation

scripts/                       # offline pipeline — at repo ROOT, not inside Qmes/
├── clf/
│   ├── 1_extract.py           # extract meta-features → meta_dataset_clf.csv
│   ├── 2_evaluate.py          # run oracle → pivot_mcc.csv
│   ├── 3_train.py             # LOO-CV across classifiers × feature subsets
│   ├── 4_select_save.py       # pick best model, save recommender bundle
│   ├── 5_inference.py         # evaluate recommender on held-out datasets
│   └── 6_baseline.py          # Wilcoxon vs LOO best-average baseline
└── reg/                       # same 6 steps for regression

datasets/                      # sample CSV datasets
results/ , backup/             # cached meta-datasets and score pivots
pyproject.toml
```

---

## Design notes

- **MCC over accuracy** for the classification Oracle — the two diverge sharply on imbalanced datasets.
- **R² over RMSE/MAE** for regression — scale-independent, so circuit rankings are comparable across datasets with different target scales.
- **PCA is fit inside the CV loop on the train split only** — fitting on full data before the split would leak information.
- **Fixed-kernel Oracle, not VQC** — a trainable ansatz would entangle encoding quality with optimization, making meta-labels unreliable.
- **Known limitation:** MI feature selection is computed before the LOO loop, which introduces a small optimistic bias. Documented and accepted for now.

---

## Roadmap


---

## Theoretical grounding

