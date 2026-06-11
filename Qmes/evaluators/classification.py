"""Qmes/oracle/classification.py

Oracle cho binary classification: quantum kernel + SVC.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from Qmes.circuits.registry import (
    UNIT_RANGE_CIRCUITS,
    compute_kernel_matrix,
    get_circuit_fn,
)
from Qmes.evaluators.base import BaseEvaluator


class ClassificationEvaluator(BaseEvaluator):

    task_type = "classification"
    metric_name = "MCC"

    def __init__(
        self,
        n_splits: int = 3,
        max_features: int = 4,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.max_features = max_features
        self.random_state = random_state

    def evaluate_circuit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        circuit_name: str,
        **kwargs,
    ) -> dict[str, float]:
        """3-fold stratified CV with SVC(precomputed kernel).

        Returns
        -------
        dict with keys: mean_acc, std_acc, mean_mcc, std_mcc, mean_f1, std_f1
        """
        circuit_fn = get_circuit_fn(circuit_name)

        if circuit_name in UNIT_RANGE_CIRCUITS:
            feature_range = (0, 1)
        else:
            feature_range = (0, np.pi)

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        acc_scores, mcc_scores, f1_scores = [], [], []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # StandardScaler → PCA → MinMaxScaler
            std_scaler = StandardScaler()
            X_train = std_scaler.fit_transform(X_train)
            X_test = std_scaler.transform(X_test)

            n_qubits = min(X_train.shape[1], self.max_features)
            if X_train.shape[1] > n_qubits:
                pca = PCA(n_components=n_qubits, random_state=self.random_state)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            minmax = MinMaxScaler(feature_range=feature_range)
            X_train = minmax.fit_transform(X_train)
            X_test = minmax.transform(X_test)

            if circuit_name in UNIT_RANGE_CIRCUITS:
                X_train = np.clip(X_train, 0, 1)
                X_test = np.clip(X_test, 0, 1)

            # Quantum kernel
            K_train = compute_kernel_matrix(X_train, X_train, circuit_fn)
            K_test = compute_kernel_matrix(X_test, X_train, circuit_fn)

            # SVC
            model = SVC(kernel="precomputed", C=1.0)
            model.fit(K_train, y_train)
            y_pred = model.predict(K_test)

            acc_scores.append(accuracy_score(y_test, y_pred))
            mcc_scores.append(matthews_corrcoef(y_test, y_pred))
            f1_scores.append(
                f1_score(y_test, y_pred, average="binary", zero_division=0)
            )

        return {
            "mean_acc": np.mean(acc_scores),
            "std_acc": np.std(acc_scores),
            "mean_mcc": np.mean(mcc_scores),
            "std_mcc": np.std(mcc_scores),
            "mean_f1": np.mean(f1_scores),
            "std_f1": np.std(f1_scores),
        }


def filter_degenerate_datasets(
    pivot: "pd.DataFrame",
    min_max_score: float = 0.1,
    ceiling_threshold: float = 0.99,
) -> tuple["pd.DataFrame", dict[str, str]]:
    """Remove no-signal and ceiling datasets from pivot table.

    Parameters
    ----------
    pivot : DataFrame, index=circuits, columns=datasets
    min_max_score : drop if max score across circuits < this
    ceiling_threshold : drop if min score across circuits >= this

    Returns
    -------
    (clean_pivot, removed_dict)
    """
    removed = {}
    for ds in pivot.columns:
        col = pivot[ds]
        if col.max() < min_max_score:
            removed[ds] = f"no-signal (max={col.max():.4f})"
        elif col.min() >= ceiling_threshold:
            removed[ds] = f"ceiling (min={col.min():.4f})"

    clean = pivot.drop(columns=list(removed.keys()))
    return clean, removed