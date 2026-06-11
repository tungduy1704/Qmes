"""Qmes/oracle/base.py

Abstract base class cho Oracle — đánh giá circuit performance trên dataset.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from Qmes.circuits.registry import CIRCUIT_POOL, get_circuit_names

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Evaluate all circuits on a dataset, return performance scores."""

    @property
    @abstractmethod
    def task_type(self) -> str:
        ...

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Primary metric name, e.g. 'MCC', 'R2'."""
        ...

    @abstractmethod
    def evaluate_circuit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        circuit_name: str,
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate one circuit on one dataset.

        Returns
        -------
        dict with keys like 'mean_mcc', 'std_mcc', etc.
        """
        ...

    def evaluate_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        circuit_names: list[str] | None = None,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        """Evaluate all circuits on one dataset.

        Returns
        -------
        {circuit_name: {metric: value, ...}, ...}
        """
        if circuit_names is None:
            circuit_names = get_circuit_names()

        results = {}
        for name in circuit_names:
            try:
                results[name] = self.evaluate_circuit(X, y, name, **kwargs)
            except Exception:
                logger.exception("Failed on circuit '%s'", name)
                results[name] = {}
        return results

    def build_pivot(
        self,
        datasets: dict[str, tuple[np.ndarray, np.ndarray]],
        circuit_names: list[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Run Oracle on all datasets × all circuits → pivot table.

        Returns
        -------
        DataFrame: index=circuit_names, columns=dataset_names,
                   values=primary metric score
        """
        if circuit_names is None:
            circuit_names = get_circuit_names()

        metric_key = f"mean_{self.metric_name.lower()}"
        records = {}

        for ds_name, (X, y) in datasets.items():
            logger.info("Oracle: %s | %s", ds_name, X.shape)
            scores = self.evaluate_all(X, y, circuit_names, **kwargs)
            records[ds_name] = {
                c: scores[c].get(metric_key, np.nan)
                for c in circuit_names
            }

        pivot = pd.DataFrame(records)  # index=circuits, columns=datasets
        pivot.index.name = "circuit"
        return pivot