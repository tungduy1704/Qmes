"""Qmes/evaluators/regression.py

Oracle for tabular regression: quantum fidelity kernel + KernelRidge, metric R²
through K-fold CV.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Qmes.circuits.registry import (
    UNIT_RANGE_CIRCUITS,
    compute_kernel_matrix,
    get_circuit_fn,
)
from Qmes.evaluators.base import BaseEvaluator 

logger = logging.getLogger(__name__)


class RegressionEvaluator(BaseEvaluator):
    """Evaluate encoding circuits for tabular regression via quantum-kernel KRR."""

    task_type = "regression"
    metric_name = "R2"  

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
        """K-fold CV (default n_splits=3) with KernelRidge(precomputed quantum kernel).

        Returns
        -------
        dict keys: mean_r2, std_r2, mean_rmse, std_rmse, mean_mae, std_mae
        """
        circuit_fn = get_circuit_fn(circuit_name)

        if circuit_name in UNIT_RANGE_CIRCUITS:
            feature_range = (0, 1)
        else:
            feature_range = (0, np.pi)

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        r2_scores, rmse_scores, mae_scores = [], [], []

        for train_idx, test_idx in kf.split(X):
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

            y_scaler = StandardScaler()
            y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

            model = KernelRidge(kernel="precomputed", alpha=1.0)
            model.fit(K_train, y_train_s)
            y_pred_s = model.predict(K_test)
            y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

            r2_scores.append(r2_score(y_test, y_pred))
            rmse_scores.append(float(np.sqrt(mean_squared_error(y_test, y_pred))))
            mae_scores.append(mean_absolute_error(y_test, y_pred))

        return {
            "mean_r2": float(np.mean(r2_scores)),
            "std_r2": float(np.std(r2_scores)),
            "mean_rmse": float(np.mean(rmse_scores)),
            "std_rmse": float(np.std(rmse_scores)),
            "mean_mae": float(np.mean(mae_scores)),
            "std_mae": float(np.std(mae_scores)),
        }