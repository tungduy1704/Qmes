"""Qmes/inference/runner.py

End-to-end inference: extract → recommend → (optional) evaluate ground truth.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from Qmes.extractors.base import BaseExtractor
from Qmes.recommender.pairwise import PairwiseRecommender
from Qmes.evaluators.base import BaseEvaluator
from Qmes.data.preprocessing import encode_categoricals, impute_and_cast
from Qmes.config import MAX_SAMPLES, TIED_THRESHOLD

logger = logging.getLogger(__name__)

def preprocess_new_dataset(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    max_samples: int = MAX_SAMPLES,
    random_state: int = 42,
    stratify: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess a new dataset to match meta-dataset conventions.

    Steps (same as data loaders):
        1. Encode categoricals → numeric
        2. Impute missing values + cast to float64
        3. Subsample to max_samples if needed

    Parameters
    ----------
    X : feature matrix (DataFrame or ndarray)
    y : target vector
    max_samples : cap sample count (default 600, see Qmes.config.MAX_SAMPLES)

    Returns
    -------
    (X_clean, y_clean) : ndarray, ready for extractor + evaluator
    """
    # Convert to DataFrame if needed (for encode/impute)
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    if isinstance(y, pd.Series):
        y = y.values

    # 1. Encode categoricals
    X = encode_categoricals(X)

    # 2. Impute + cast
    X = impute_and_cast(X)

    # 3. To ndarray
    X = X.values.astype(np.float64)
    y = y.astype(np.float64)

    # 4. Subsample
    if len(X) > max_samples:
        if stratify:
            from sklearn.model_selection import train_test_split
            _, X, _, y = train_test_split(
                X, y,
                test_size=max_samples,
                stratify=y,
                random_state=random_state,
            )
        else:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(len(X), size=max_samples, replace=False)
            X = X[idx]
            y = y[idx]

    return X, y

def recommend(
    X,
    y,
    extractor: BaseExtractor,
    recommender: PairwiseRecommender,
    top_k: int = 3,
    preprocess: bool = True,
    stratify: bool = False,
) -> dict:
    """Recommend quantum encoding circuits for a new dataset.

    The end-to-end inference entry point: extracts meta-features from
    (X, y), queries the pre-trained recommender, and returns a ranked
    list of circuits — no quantum evaluation at inference time.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix of the new dataset.
    y : array-like of shape (n_samples,)
        Target vector.
    extractor : BaseExtractor
        Fitted or stateless extractor matching the task type, e.g.
        ``get_extractor('classification')``.
    recommender : PairwiseRecommender
        Pre-trained recommender, e.g. from ``load_default_recommender()``.
        Must have the same task_type as *extractor*.
    top_k : int, default=3
        Number of top circuits to return.
    preprocess : bool, default=True
        If True, apply standard preprocessing (scaling, imputation)
        before meta-feature extraction.
    stratify : bool, default=False
        If True, use stratified splitting during preprocessing.
        Relevant for classification tasks with imbalanced classes.

    Returns
    -------
    dict with keys:
        'ranking' : list[str]
            All 7 circuits sorted by votes (descending).
        'top_k' : list[str]
            Top-k circuits (first *top_k* elements of 'ranking').
        'votes' : dict[str, int]
            Raw OvO vote counts per circuit.
        'meta_features' : np.ndarray of shape (d,)
            Extracted meta-feature vector used for this recommendation.

    Raises
    ------
    ValueError
        If extractor.task_type != recommender.task_type.

    Examples
    --------
    >>> from Qmes import get_extractor, load_default_recommender, recommend
    >>> rec = load_default_recommender('classification')
    >>> ext = get_extractor('classification')
    >>> result = recommend(X, y, extractor=ext, recommender=rec)
    >>> result['top_k']
    ['RY', 'HERx', 'ZFM']
    """
    
    if preprocess:
        X, y = preprocess_new_dataset(X, y, stratify=stratify)

    result = extractor.extract(X, y)

    rec_task = getattr(recommender, "task_type", None)
    if rec_task is not None and result.task_type is not None and rec_task != result.task_type:
        raise ValueError(
            f"Task-type mismatch giữa extractor và recommender:\n"
            f"  extractor  : {result.task_type}\n"
            f"  recommender: {rec_task}"
        )

    expected = getattr(recommender, "feature_names", None)
    if expected is not None and list(result.feature_names) != list(expected):
        raise ValueError(
            f"Feature-name mismatch giữa extractor và recommender:\n"
            f"  extractor : {result.feature_names}\n"
            f"  recommender: {expected}"
        )
    pred = recommender.predict(result.vector, top_k=top_k)
    pred["meta_features"] = result.vector
    return pred

def evaluate_recommendation(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    extractor: BaseExtractor,
    recommender: PairwiseRecommender,
    evaluator: BaseEvaluator,
    top_k: int = 3,
    tied_threshold: float = TIED_THRESHOLD,
    preprocess: bool = True,
) -> pd.DataFrame:
    """Run full inference evaluation on multiple datasets.

    For each dataset:
    1. Recommend circuits (no quantum eval)
    2. Run Oracle ground truth (quantum eval)
    3. Compare

    Returns
    -------
    DataFrame with per-dataset results
    """
    rec_task = getattr(recommender, "task_type", None)
    if rec_task is not None and rec_task != evaluator.task_type:
        raise ValueError(
            f"Task-type mismatch giữa recommender và evaluator:\n"
            f"  recommender: {rec_task}\n"
            f"  evaluator  : {evaluator.task_type}"
        )

    metric_key = f"mean_{evaluator.metric_name.lower()}"
    rows = []

    for ds_name, (X, y) in datasets.items():
        logger.info("Inference: %s | %s", ds_name, X.shape)

        # Preprocess ONCE
        if preprocess:
            X, y = preprocess_new_dataset(X, y)

        # 1. Recommend
        rec = recommend(X, y, extractor, recommender, top_k=top_k, preprocess=False)

        # 2. Ground truth from Oracle
        oracle_scores = evaluator.evaluate_all(X, y)
        score_dict = {
            c: oracle_scores[c].get(metric_key, np.nan)
            for c in oracle_scores
        }

        valid_scores = {c: s for c, s in score_dict.items() if not np.isnan(s)}
        if not valid_scores:
            logger.warning("%s: all circuits failed, skipping", ds_name)
            continue
        if len(valid_scores) < len(score_dict):
            logger.warning("%s: dropped %d failed circuits",
                        ds_name, len(score_dict) - len(valid_scores))
            
        # 3. Compare
        true_ranking = sorted(valid_scores, key=valid_scores.get, reverse=True)
        true_best = true_ranking[0]
        best_score = valid_scores[true_best]

        # Tied set
        tied_set = {
            c for c, s in valid_scores.items()
            if s >= best_score - tied_threshold
        }

        rec_top1 = rec["top_k"][0]
        rec_top1_score = valid_scores.get(rec_top1, np.nan)

        rows.append({
            "dataset": ds_name,
            "rec_top1": rec_top1,
            "rec_top_k": rec["top_k"],
            "true_best": true_best,
            "true_top3": true_ranking[:3],
            "tied_hit": rec_top1 in tied_set,
            "top3_tied_hit": any(c in tied_set for c in rec["top_k"]),
            "regret": round(best_score - rec_top1_score, 4),
            "best_score": round(best_score, 4),
            "rec_score": round(rec_top1_score, 4),
        })

    df = pd.DataFrame(rows)

    # Summary
    n = len(df)
    if n > 0:
        logger.info(
            "Tied=%.3f, Top3_Tied=%.3f, Mean_Regret=%.4f",
            df["tied_hit"].mean(),
            df["top3_tied_hit"].mean(),
            df["regret"].mean(),
        )

    return df

