"""Qmes/recommender/selection.py

Model selection: LOO evaluation, MI feature selection, classifier pool.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    AdaBoostClassifier, BaggingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from Qmes.config import TIED_THRESHOLD


# ── Classifier pool ──────────────────────────────────────────────────────────

DEFAULT_CLASSIFIERS = {
    "DT":          DecisionTreeClassifier(random_state=42),
    "RF":          RandomForestClassifier(n_estimators=100, random_state=42),
    "GB":          GradientBoostingClassifier(random_state=42),
    "AdaBoost":    AdaBoostClassifier(
                       estimator=DecisionTreeClassifier(random_state=42),
                       random_state=42),
    "Bagging":     BaggingClassifier(random_state=42),
    "SVM-linear":  SVC(kernel="linear",  probability=True, random_state=42),
    "SVM-rbf":     SVC(kernel="rbf",     probability=True, random_state=42),
    "SVM-sigmoid": SVC(kernel="sigmoid", probability=True, random_state=42),
    "MLP-small":   MLPClassifier(hidden_layer_sizes=(64, 32),
                                 max_iter=500, random_state=42),
    "MLP-large":   MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                                 max_iter=500, random_state=42),
    "kNN":         KNeighborsClassifier(),
    "NearCentroid": NearestCentroid(),
    "NaiveBayes":  GaussianNB(),
    "LogReg":      LogisticRegression(max_iter=1000, random_state=42),
}


# ── Feature selection ────────────────────────────────────────────────────────

def select_features_mi(
    meta_features: np.ndarray,
    labels: np.ndarray,
    k_values: list[int] = (5, 10, 15, 20),
    random_state: int = 42,
) -> dict[str, list[int]]:
    """Rank meta-features by mutual information and return index subsets.

    MI is computed between each meta-feature and `labels` (the best circuit
    per dataset — a categorical target), then features are ranked
    descending. Used at training time to test how few meta-features still
    yield a good recommender.

    Parameters
    ----------
    meta_features : ndarray, shape (n_datasets, d)
        Meta-feature matrix, one row per dataset.
    labels : ndarray, shape (n_datasets,)
        Categorical target — the best circuit for each dataset, e.g.
        ``pivot_scores.idxmax(axis=0)``. MI is computed via
        mutual_info_classif, so this must be the circuit label, NOT the
        dataset's own y.
    k_values : tuple[int, ...], default=(5, 10, 15, 20)
        Subset sizes to produce. A size is skipped if it exceeds d.
    random_state : int, default=42
        Seed for mutual_info_classif (its k-NN estimator is stochastic).

    Returns
    -------
    dict[str, list[int]]
        Index subsets keyed by label:
        {"full": all d indices, "top5": [...], "top10": [...], ...},
        each a list of column indices into meta_features ordered by
        descending MI.
    """
    mi_scores = mutual_info_classif(meta_features, labels, random_state=random_state)
    ranked = np.argsort(mi_scores)[::-1]

    subsets = {"full": list(range(meta_features.shape[1]))}
    for k in k_values:
        if k <= meta_features.shape[1]:
            subsets[f"top{k}"] = ranked[:k].tolist()

    return subsets


# ── LOO evaluation ───────────────────────────────────────────────────────────

def run_loo_evaluation(
    meta_features: np.ndarray,
    pivot_scores: pd.DataFrame,
    classifiers: dict | None = None,
    feature_subsets: dict[str, list[int]] | None = None,
    tied_threshold: float = TIED_THRESHOLD,
    verbose: bool = True,
    dataset_names: list[str] | None = None,
) -> pd.DataFrame:
    """Exhaustive LOO evaluation over (classifier × feature_subset).

    Parameters
    ----------
    meta_features : (n_datasets, d)
        Row i MUST correspond to ``pivot_scores.columns[i]``. Alignment is
        positional: this function never joins by name (meta_features is a bare
        array with no labels). Callers are responsible for passing meta rows in
        pivot-column order, e.g. ``meta.loc[pivot.columns].values``.
    pivot_scores : index=circuits, columns=datasets
    classifiers : dict {name: sklearn_estimator}
    feature_subsets : dict {label: list_of_indices}
    tied_threshold : for tied-best evaluation
    verbose : print progress
    dataset_names : list[str] or None
        Optional alignment guard. If given, must be the dataset name for each
        meta-feature row IN ROW ORDER; the function asserts it equals
        ``list(pivot_scores.columns)`` and raises ValueError otherwise. Pass
        ``list(meta.index)`` to turn the positional-alignment invariant from a
        convention the caller must remember into a contract the library checks.

    Returns
    -------
    DataFrame with columns: Features, Classifier, Single, Tied, Top3_Tied, Mean_Regret
    """
    # ── alignment guards ─────────────────────────────────────────────────
    # Always-on shape check: catches the grossest misalignment (wrong number
    # of meta rows) regardless of whether names are supplied.
    if meta_features.shape[0] != pivot_scores.shape[1]:
        raise ValueError(
            f"meta_features has {meta_features.shape[0]} rows but pivot has "
            f"{pivot_scores.shape[1]} datasets — they must match one-to-one."
        )
    # Opt-in name check: the real silent-corruption guard.
    if dataset_names is not None and list(dataset_names) != list(pivot_scores.columns):
        raise ValueError(
            "Dataset alignment broken: meta-feature row order does not match "
            "pivot column order. run_loo_evaluation pairs meta row i with "
            "pivot column i BY POSITION, so a mismatch silently trains each "
            "classifier on the wrong labels.\n"
            f"  dataset_names[:3] = {list(dataset_names)[:3]}\n"
            f"  pivot.columns[:3] = {list(pivot_scores.columns)[:3]}\n"
            "Fix: pass meta reindexed to pivot order, e.g. meta.loc[pivot.columns]."
        )

    if classifiers is None:
        classifiers = DEFAULT_CLASSIFIERS
    if feature_subsets is None:
        y_single = pivot_scores.idxmax(axis=0).values
        feature_subsets = select_features_mi(meta_features, y_single)

    circuits = list(pivot_scores.index)
    datasets = list(pivot_scores.columns)
    pairs = list(combinations(circuits, 2))
    n = len(datasets)
    loo = LeaveOneOut()

    # Precompute
    y_single = pivot_scores.idxmax(axis=0).values
    tied_sets = {
        ds: set(pivot_scores[ds][
            pivot_scores[ds] >= pivot_scores[ds].max() - tied_threshold
        ].index)
        for ds in datasets
    }

    # Pairwise labels
    pairwise_y = {}
    for (c1, c2) in pairs:
        pairwise_y[(c1, c2)] = np.array([
            1 if pivot_scores.loc[c1, ds] >= pivot_scores.loc[c2, ds] else 0
            for ds in datasets
        ])

    # Evaluate
    rows = []
    for feat_label, feat_idx in feature_subsets.items():
        if verbose:
            print(f"\n=== [{feat_label}] ===")

        X_sub = meta_features[:, feat_idx]

        for model_name, model in classifiers.items():
            single_correct = tied_correct = top3_tied = 0
            total_regret = 0.0

            for train_idx, test_idx in loo.split(meta_features):
                test_ds = datasets[test_idx[0]]
                votes = {c: 0 for c in circuits}

                for (c1, c2) in pairs:
                    clf = clone(model)
                    clf.fit(X_sub[train_idx], pairwise_y[(c1, c2)][train_idx])
                    pred = clf.predict(X_sub[test_idx])[0]
                    if pred == 1:
                        votes[c1] += 1
                    else:
                        votes[c2] += 1

                mv_pred = max(votes, key=votes.get)
                ranked = sorted(votes, key=votes.get, reverse=True)[:3]

                if mv_pred == y_single[test_idx[0]]:
                    single_correct += 1
                if mv_pred in tied_sets[test_ds]:
                    tied_correct += 1
                top3_tied += int(any(c in tied_sets[test_ds] for c in ranked))
                total_regret += (
                    pivot_scores[test_ds].max() - pivot_scores.loc[mv_pred, test_ds]
                )

            result = {
                "Features":    feat_label,
                "Classifier":  model_name,
                "Single":      round(single_correct / n, 4),
                "Tied":        round(tied_correct / n, 4),
                "Top3_Tied":   round(top3_tied / n, 4),
                "Mean_Regret": round(total_regret / n, 6),
            }
            rows.append(result)

            if verbose:
                print(f"  [{model_name}] Tied={result['Tied']} "
                      f"Top3={result['Top3_Tied']} "
                      f"Regret={result['Mean_Regret']}")

    return pd.DataFrame(rows)