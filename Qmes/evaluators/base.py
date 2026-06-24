"""Qmes/evaluators/base.py

Abstract base class for task-specific Oracles plus the task-agnostic
degenerate-dataset filter. Concrete evaluators score every circuit on a
dataset via cross-validated quantum-kernel methods (offline only).
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


def filter_degenerate_datasets(
    pivot: pd.DataFrame,
    min_max_score: float = 0.1,
    ceiling_threshold: float = 0.99,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Remove no-signal and ceiling datasets from a pivot table.

    Task-agnostic: operates on any (circuit x dataset) score pivot
    regardless of whether scores are MCC, R2, or another metric, so it
    lives here rather than in a concrete task's evaluator module.

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