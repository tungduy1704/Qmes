# Validation

Two separate questions, validated separately: does the software do what it
says (correctness), and are its recommendations actually good (scientific
validity)?

## Software correctness

The package ships **48 tests across five files**, run in CI on Python 3.10,
3.11, and 3.12, plus a minimum-dependency job that verifies the declared
version floors (`numpy 1.21`, `pandas 1.3`, `scikit-learn 1.3`). All
fixtures are synthetic, so the suite needs no network access;
`test_evaluators.py` nonetheless runs the real Qsun quantum kernel rather
than a stub.

Beyond shape and type checks, several tests target behavior:

- A fixture plants a known ground-truth circuit per dataset group and
  asserts `PairwiseRecommender` recovers the assignment after fitting -
  the OvO decomposition learns the intended signal, not just
  correctly-shaped output.
- A task-type mismatch between extractor and recommender must raise an
  explicit error, never silently produce a meaningless recommendation.
- A regression guard permutes meta-feature rows against the pivot columns
  and asserts performance degrades - `run_loo_evaluation` pairs rows to
  columns positionally, and this test proves the alignment matters.
- On a noise-only meta-dataset, the structural invariant
  `Single ≤ Tied ≤ Top3_Tied` must hold on every LOO row.

## Scientific validity

Recommendations are scored by **regret**: the per-dataset gap between the
best achievable circuit score and the score of the recommended circuit,

$$
\rho_i = s_{i,k^*_i} - s_{i,\hat{k}_i},
$$

which - unlike top-1 accuracy - penalizes a near-miss less than a wide
miss, and is well-defined even when several circuits tie for best.

Qmes is compared against three baselines under the same leave-one-out
constraint (dataset $i$'s own scores are never used to recommend for $i$):

| Baseline | Recommends |
|---|---|
| LOO Best-Avg | circuit with highest mean score over the other $N-1$ datasets |
| LOO Modal | circuit most frequently ranked best among the other $N-1$ |
| Random | expected regret of a uniform random choice |

None of the baselines consults the dataset's meta-features, isolating the
value of complexity-conditioned recommendation.

### Results

| Task | $N$ | Best-Avg $\bar\rho$ | Qmes $\bar\rho$ | Reduction |
|---|---|---|---|---|
| Classification | 105 | 0.0366 | **0.0183** | 2.0× |
| Regression | 86 | 0.0626 | **0.0150** | 4.2× |

For regression, LOO Modal is the tighter baseline ($\bar\rho = 0.0570$);
against it Qmes's reduction is still 3.8×. A paired Wilcoxon signed-rank
test on per-dataset regret differences confirms the improvement is
systematic rather than driven by a few favorable datasets
($p < 10^{-4}$ for both tasks).

### The shipped default recommenders

Model selection searched [14 classifiers](#the-classifier-search-grid) ×
MI-selected feature subsets by exhaustive LOO (see
[Recommender](api/recommender.md)). The selected configurations, refit on
the full meta-dataset and shipped in `Qmes/_models/`:

| Task | Config | Meta-features used | LOO regret | Top-3 tied acc. |
|---|---|---|---|---|
| Classification | kNN, top-5 MI | `n4`, `l3`, `f1v`, `l2`, `density` | 0.0183 | 0.905 |
| Regression | kNN, top-10 MI | `c1`, `c3`, `l1`, `l3`, `l2`, `c4`, `s4`, `c2`, `s2`, `s3` | 0.0150 | 0.965 |

For classification, kNN top-10 achieved marginally lower regret (0.0165)
but substantially lower top-3 tied accuracy (0.848 vs 0.905), so top-5 was
selected. See [Meta-features](meta_features.md) for what each measure is.

### The classifier search grid

The 14 base classifiers searched during model selection, each paired with
every MI-selected feature subset. All are
[scikit-learn](https://scikit-learn.org/stable/supervised_learning.html)
estimators at the parameters listed here; the winning configuration per
task is in the table above.

| Category | Classifier | Abbr. | Parameters |
|---|---|---|---|
| Tree-based | Decision Tree | DT | `max_depth=None` |
| Tree-based | Random Forest | RF | `n_estimators=10` |
| Ensemble | Gradient Boosting | E-GB | `n_estimators=100` |
| Ensemble | AdaBoost | AB | `n_estimators=50` |
| Ensemble | Bagging | Bg | `n_estimators=10` |
| SVM | SVM-Linear | SVM-L | `kernel='linear'` |
| SVM | SVM-RBF | SVM-R | `kernel='rbf'`, `C=1.0` |
| SVM | SVM-Sigmoid | SVM-S | `kernel='sigmoid'` |
| Neural network | MLP (500) | MLP-1 | `hidden=(500,)` |
| Neural network | MLP (100-100-100) | MLP-3 | `hidden=(100,100,100)` |
| Instance-based | k-NN | KNN | `n_neighbors=5` |
| Instance-based | Nearest Centroid | NC | `metric='euclidean'` |
| Probabilistic | Naive Bayes | NB | Gaussian |
| Probabilistic | Logistic Regression | LR | `max_iter=1000` |
