"""Qmes/data/reg/__init__.py
Regression benchmark dataset loaders (internal to scripts/reg pipeline).

Not part of the top-level Qmes public API — these load the specific UCI/
synthetic datasets used to train and evaluate the bundled recommender, not
a general-purpose dataset utility.
"""
from Qmes.data.reg.train import load_regression_datasets
from Qmes.data.reg.inference import load_inference_reg_datasets

__all__ = [
    "load_regression_datasets",
    "load_inference_reg_datasets",
]