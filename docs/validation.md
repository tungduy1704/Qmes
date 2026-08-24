# Validation

Two separate questions, validated separately: does the software do what it
says (correctness), and are its recommendations actually good (scientific
validity)?

## Scientific validity

Recommendations are scored by **regret**: the per-dataset gap between the
best achievable circuit score and the score of the recommended circuit,

$$
\rho_i = \bigl(\mathcal{Y}_{i,k^*_i} - \mathcal{Y}_{i,\hat{k}_i}\bigr),
$$

which - unlike top-1 accuracy - penalizes a near-miss less than a wide
miss, and is well-defined even when several circuits tie for best.

Qmes is compared against three baselines under the same leave-one-out
constraint:

| Baseline | Recommends |
|---|---|
| LOO Best-Avg | circuit with highest mean score |
| LOO Modal | circuit that is single best on most datasets |
| Random | expected regret of a uniform random choice |

None of the baselines consults the dataset's meta-features, isolating the
value of complexity-conditioned recommendation.

### Results

| Task | $N$ | Best-Avg $\bar\rho$ | Qmes $\bar\rho$ | Reduction |
|---|---|---|---|---|
| Classification | 105 | 0.0366 | **0.0165** | 2.2× |
| Regression | 86 | 0.0626 | **0.0150** | 4.2× |

### The shipped default recommenders

Model selection searched [14 classifiers](#the-classifier-search-grid) ×
MI-selected feature subsets by exhaustive LOO (see
[Recommender](api/recommender.md)). The selected configurations, refit on
the full meta-dataset and shipped in `Qmes/_models/`:

| Task | Config | Meta-features used | LOO regret |
|---|---|---|---|
| Classification | kNN, top-10 MI | `n4`, `l3`, `f1v`, `l2`, `density`, `lsc`, `t2`, `cls_coef`, `f1`, `t1` | 0.0165 |
| Regression | kNN, top-10 MI | `c1`, `c3`, `l1`, `l3`, `l2`, `c4`, `s4`, `c2`, `s2`, `s3` | 0.0150 |

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
