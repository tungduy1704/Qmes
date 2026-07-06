"""Qmes/evaluators/__init__.py

Task-specific Oracle evaluation (quantum kernel methods).
"""
from Qmes.evaluators.base import BaseEvaluator, filter_degenerate_datasets
from Qmes.evaluators.classification import ClassificationEvaluator
from Qmes.evaluators.regression import RegressionEvaluator

_REGISTRY: dict[str, type[BaseEvaluator]] = {
    "classification": ClassificationEvaluator,
    "regression": RegressionEvaluator,
}

def get_evaluator(task_type: str, **kwargs) -> BaseEvaluator:
    """Return the Oracle evaluator for the given task type.

    Parameters
    ----------
    task_type : {'classification', 'regression'}
    **kwargs : passed to evaluator constructor (e.g. n_splits, max_features)

    Returns
    -------
    Concrete BaseEvaluator instance
    """
    key = task_type.lower().replace("-", "").replace("_", "")

    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown task type '{task_type}'. Available: {available}"
        )

    return _REGISTRY[key](**kwargs)

__all__ = [
    "BaseEvaluator",
    "ClassificationEvaluator",
    "RegressionEvaluator",
    "filter_degenerate_datasets",
    "get_evaluator",
]