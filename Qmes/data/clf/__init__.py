"""Qmes/data/clf/__init__.py

Classification benchmark dataset loaders (internal to scripts/clf pipeline).

Not part of the top-level Qmes public API — these load the specific UCI/
synthetic datasets used to train and evaluate the bundled recommender, not
a general-purpose dataset utility.
"""
from Qmes.data.clf.train import load_classification_datasets
from Qmes.data.clf.inference import load_inference_classification

__all__ = [
    "load_classification_datasets",
    "load_inference_classification",
]