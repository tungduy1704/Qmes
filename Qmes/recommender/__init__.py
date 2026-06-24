"""Qmes/recommender/__init__.py

Pairwise meta-learner: recommends circuits from meta-features.
"""
import importlib.resources

from Qmes.config import TIED_THRESHOLD
from Qmes.recommender.pairwise import PairwiseRecommender
from Qmes.recommender.selection import (
    DEFAULT_CLASSIFIERS,
    select_features_mi,
    run_loo_evaluation,
)

# task_type -> default metric_name. Manual duplicate of
# ClassificationEvaluator.metric_name / RegressionEvaluator.metric_name —
# keep in sync by hand if those change.
_DEFAULT_METRIC_NAME = {
    "classification": "MCC",
    "regression": "R2",
}

def get_recommender(
    task_type: str,
    classifier,
    feature_indices: list[int] | None = None,
    tied_threshold: float = TIED_THRESHOLD,
    feature_names: list[str] | None = None,
    metric_name: str | None = None,
) -> PairwiseRecommender:
    """Build a PairwiseRecommender with task_type/metric_name auto-filled.

    There is only one recommender class (PairwiseRecommender is task-
    agnostic: behavior depends on the `classifier` and data passed in, not
    on task_type), so this factory exists mainly to remove a manual-typo
    failure mode -- passing the wrong metric_name for a task_type used to
    be possible by hand and would only surface as a confusing label later.

    Parameters
    ----------
    task_type : {'classification', 'regression'}
    classifier : sklearn estimator (unfitted template), still supplied by
        the caller -- there is no per-task default classifier.
    feature_indices : list[int] or None, default=None
        Which meta-feature columns the recommender uses. None = all.
    tied_threshold : float, default=TIED_THRESHOLD
        Tie tolerance for downstream evaluation; stored and persisted on
        the recommender, not applied by predict() itself.
    feature_names : list[str] or None, default=None
        Full ordered meta-feature names expected from the extractor (before
        feature_indices subsetting). Persisted so inference can assert order
        alignment after load().
    metric_name : str or None, default=None
        Override the auto-filled default ('MCC' for classification, 'R2'
        for regression) if needed.

    Returns
    -------
    PairwiseRecommender (unfitted; call .fit(X, pivot) before saving)
    """
    key = task_type.lower().replace("-", "").replace("_", "")
    if key not in _DEFAULT_METRIC_NAME:
        available = ", ".join(sorted(_DEFAULT_METRIC_NAME.keys()))
        raise ValueError(
            f"Unknown task type '{task_type}'. Available: {available}"
        )

    return PairwiseRecommender(
        classifier=classifier,
        feature_indices=feature_indices,
        tied_threshold=tied_threshold,
        feature_names=feature_names,
        task_type=key,
        metric_name=metric_name or _DEFAULT_METRIC_NAME[key],
    )


def load_default_recommender(task_type: str) -> PairwiseRecommender:
    """Load the pre-trained recommender bundled with the Qmes package.

    This is the entry point for the package's core value proposition:
    pip install Qmes -> recommend a circuit -> no quantum evaluation
    needed at inference time. The bundle is shipped as package data
    (see [tool.setuptools.package-data] in pyproject.toml) under
    Qmes/_models/<task_type>/, not regenerated at import time.

    Parameters
    ----------
    task_type : {'classification', 'regression'}

    Returns
    -------
    PairwiseRecommender, already fitted, ready for .predict() or for
    Qmes.recommend(X, y, extractor, recommender=...).

    Raises
    ------
    ValueError : unknown task_type
    FileNotFoundError : package was installed without the bundled model
        data (e.g. a from-source checkout where Qmes/_models/<task_type>/
        was never populated -- this is a packaging/build issue, not a
        normal runtime condition).
    """
    key = task_type.lower().replace("-", "").replace("_", "")
    if key not in _DEFAULT_METRIC_NAME:
        available = ", ".join(sorted(_DEFAULT_METRIC_NAME.keys()))
        raise ValueError(
            f"Unknown task type '{task_type}'. Available: {available}"
        )

    model_dir = importlib.resources.files("Qmes._models") / key
    if not (model_dir / "recommender.pkl").is_file():
        raise FileNotFoundError(
            f"No bundled recommender for task_type='{key}' at {model_dir}. "
            f"This Qmes install is missing its packaged model data -- "
            f"re-run scripts/{('clf' if key == 'classification' else 'reg')}/"
            f"4_select_save.py and copy the bundle into Qmes/_models/{key}/, "
            f"or reinstall the package."
        )

    # importlib.resources.files() may return a path inside a zip/wheel;
    # PairwiseRecommender.load() needs a real filesystem directory, so
    # use as_file() to materialize it if necessary (no-op for a normal
    # on-disk install).
    with importlib.resources.as_file(model_dir) as real_dir:
        return PairwiseRecommender.load(real_dir)


__all__ = [
    "PairwiseRecommender",
    "DEFAULT_CLASSIFIERS",
    "select_features_mi",
    "run_loo_evaluation",
    "get_recommender",
    "load_default_recommender",
]