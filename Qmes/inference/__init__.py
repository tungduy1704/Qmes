"""Qmes/inference/__init__.py

Inference-time pipeline: extract -> recommend -> (optionally) evaluate.
"""
from Qmes.inference.runner import (
    preprocess_new_dataset,
    recommend,
    evaluate_recommendation,
)

__all__ = [
    "preprocess_new_dataset",
    "recommend",
    "evaluate_recommendation",
]