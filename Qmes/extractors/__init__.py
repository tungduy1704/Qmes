"""qmatch/extractors — Task-specific meta-feature extraction.

Mỗi task type có một extractor riêng, kế thừa từ BaseExtractor.
Tất cả trả về ExtractionResult(vector, feature_names, task_type).

Usage:
    from qmatch.extractors import get_extractor

    ext = get_extractor("classification")
    result = ext.extract(X, y)
    print(result.vector.shape)       # (24,)
    print(result.feature_names)      # ['f1', 'f1v', ...]
    print(result.to_dict())          # {'f1': 0.82, 'f1v': 0.91, ...}
"""
from Qmes.extractors.base import BaseExtractor, ExtractionResult
from Qmes.extractors.classification import ClassificationExtractor
from Qmes.extractors.regression import RegressionExtractor
from Qmes.extractors.timeseries import TimeSeriesExtractor
from Qmes.extractors.multilabel import MultilabelExtractor
from Qmes.extractors.clustering import ClusteringExtractor
from Qmes.extractors.anomaly import AnomalyExtractor

_REGISTRY: dict[str, type[BaseExtractor]] = {
    "classification": ClassificationExtractor,
    "regression": RegressionExtractor,
    "timeseries": TimeSeriesExtractor,
    "multilabel": MultilabelExtractor,
    "clustering": ClusteringExtractor,
    "anomaly": AnomalyExtractor,
}


def get_extractor(task_type: str, **kwargs) -> BaseExtractor:
    """Factory: lấy extractor theo task type.

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
        "timeseries": "timeseries",
        "time_series": "timeseries",
        "multilabel": "multilabel",
        "multi_label": "multilabel",
        "clustering": "clustering",
        "anomaly": "anomaly",
        "anomalydetection": "anomaly",
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
    "TimeSeriesExtractor",
    "MultilabelExtractor",
    "ClusteringExtractor",
    "AnomalyExtractor",
    "get_extractor",
]