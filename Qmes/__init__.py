# Qmes/__init__.py
__version__ = "0.1.0"

from Qmes.extractors import get_extractor, ClassificationExtractor, RegressionExtractor
from Qmes.evaluators import get_evaluator, ClassificationEvaluator, RegressionEvaluator, filter_degenerate_datasets
from Qmes.recommender import get_recommender, load_default_recommender, PairwiseRecommender, run_loo_evaluation
from Qmes.inference import recommend, evaluate_recommendation, preprocess_new_dataset
from Qmes.circuits import CIRCUIT_POOL, get_circuit_names

__all__ = [
    "__version__",
    "get_extractor", "ClassificationExtractor", "RegressionExtractor",
    "get_evaluator", "ClassificationEvaluator", "RegressionEvaluator",
    "run_loo_evaluation", "filter_degenerate_datasets",
    "get_recommender", "load_default_recommender", "PairwiseRecommender",
    "recommend", "evaluate_recommendation", "preprocess_new_dataset",
    "CIRCUIT_POOL", "get_circuit_names",
]