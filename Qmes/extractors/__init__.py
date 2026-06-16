"""Task-specific meta-feature extraction."""
from Qmes.extractors.base import BaseExtractor, ExtractionResult
from Qmes.extractors.classification import ClassificationExtractor
from Qmes.extractors.regression import RegressionExtractor


_REGISTRY: dict[str, type[BaseExtractor]] = {
    "classification": ClassificationExtractor,
    "regression": RegressionExtractor,
}

def get_extractor(task_type: str, **kwargs) -> BaseExtractor:
    """Return the extractor for the given task type.

    Parameters
    ----------
    task_type : one of 'classification', 'regression', 'timeseries',
                'multilabel', 'clustering', 'anomaly'
    **kwargs : passed to extractor constructor (e.g. k_min, k_max for clustering)

    Returns
    -------
    Concrete BaseExtractor instance
    """
    key = task_type.lower().replace("-", "").replace("_", "")
    # Normalize common aliases
    aliases = {
        "classification": "classification",
        "regression": "regression",
    }
    normalized = aliases.get(key, key)

    if normalized not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown task type '{task_type}'. Available: {available}"
        )

    return _REGISTRY[normalized](**kwargs)

__all__ = [
    "BaseExtractor",
    "ExtractionResult",
    "ClassificationExtractor",
    "RegressionExtractor",
    "get_extractor",
]