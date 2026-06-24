"""Qmes/recommender/pairwise.py

Pairwise OvO recommender: trains one binary classifier per circuit pair, aggregates votes to rank circuits.
"""
from __future__ import annotations

import logging
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from Qmes.config import TIED_THRESHOLD

logger = logging.getLogger(__name__)


class PairwiseRecommender:
    """Pairwise One-vs-One circuit recommender.

    Parameters
    ----------
    classifier : sklearn estimator (unfitted template)
    feature_indices : list[int] or None
        Which meta-feature columns to use. None = all.
    tied_threshold : float
        Tie tolerance for downstream evaluation (run_loo_evaluation,
        evaluate_recommendation). 
    feature_names : list[str] or None
        Full ordered list of meta-feature names this recommender expects
        from the extractor (before feature_indices subsetting). Persisted
        so inference can assert order alignment after load().
    task_type : str or None
        E.g. 'classification' or 'regression'. Persisted so inference can
        assert the recommender matches the extractor/evaluator in use.
    metric_name : str or None
        E.g. 'MCC' or 'R2'. Stored for traceability only.
    """

    def __init__(
        self,
        classifier,
        feature_indices: list[int] | None = None,
        tied_threshold: float = TIED_THRESHOLD,
        feature_names: list[str] | None = None,
        task_type: str | None = None,
        metric_name: str | None = None,
    ):
        self.classifier = classifier
        self.feature_indices = feature_indices
        self.tied_threshold = tied_threshold
        self.feature_names = feature_names
        self.task_type = task_type
        self.metric_name = metric_name

        # Set after fit()
        self.circuits_: list[str] = []
        self.pairs_: list[tuple[str, str]] = []
        self.classifiers_: dict[tuple[str, str], object] = {}
        self.is_fitted_ = False

    def fit(
        self,
        meta_features: np.ndarray | pd.DataFrame,
        pivot_scores: pd.DataFrame,
    ) -> "PairwiseRecommender":
        """Train pairwise classifiers.

        Parameters
        ----------
        meta_features : (n_datasets, d) meta-feature matrix
            Row i must correspond to pivot_scores.columns[i]. Alignment is
            positional — if a DataFrame is passed, only its .values are used,
            not its index.
        pivot_scores : DataFrame, index=circuits, columns=datasets
            Values = primary metric (MCC, R², etc.)
        """
        if isinstance(meta_features, pd.DataFrame):
            meta_features = meta_features.values

        X = self._select_features(meta_features)
        self.circuits_ = list(pivot_scores.index)
        self.pairs_ = list(combinations(self.circuits_, 2))

        # Build pairwise labels and train
        datasets = list(pivot_scores.columns)
        self.classifiers_ = {}

        for (c1, c2) in self.pairs_:
            y_pair = np.array([
                1 if pivot_scores.loc[c1, ds] >= pivot_scores.loc[c2, ds] else 0
                for ds in datasets
            ])
            clf = clone(self.classifier)
            clf.fit(X, y_pair)
            self.classifiers_[(c1, c2)] = clf

        self.is_fitted_ = True
        logger.info(
            "Fitted %d pairwise classifiers for %d circuits",
            len(self.pairs_), len(self.circuits_),
        )
        return self

    def predict(
        self,
        meta_features: np.ndarray,
        top_k: int = 3,
    ) -> dict:
        """Predict circuit ranking for a new dataset.

        Parameters
        ----------
        meta_features : (d,) or (1, d) meta-feature vector
        top_k : number of top circuits to return

        Returns
        -------
        dict with keys:
            'ranking': list of all circuits sorted by votes (descending)
            'top_k': list of top-k circuits
            'votes': dict {circuit: vote_count}
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() first.")

        x = np.atleast_2d(meta_features)
        x = self._select_features(x)

        votes = {c: 0 for c in self.circuits_}
        for (c1, c2) in self.pairs_:
            pred = self.classifiers_[(c1, c2)].predict(x)[0]
            if pred == 1:
                votes[c1] += 1
            else:
                votes[c2] += 1

        ranking = sorted(votes, key=votes.get, reverse=True)

        return {
            "ranking": ranking,
            "top_k": ranking[:top_k],
            "votes": votes,
        }

    def save(self, path: str | Path) -> None:
        """Save fitted recommender to directory (writes recommender.pkl)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        artifact = {
            "format_version": 1,
            "task_type": self.task_type,
            "metric_name": self.metric_name,
            "feature_names": self.feature_names,
            "feature_indices": self.feature_indices,
            "tied_threshold": self.tied_threshold,
            "circuits": self.circuits_,
            "pairs": self.pairs_,
            "classifiers": self.classifiers_,
        }
        with open(path / "recommender.pkl", "wb") as f:
            pickle.dump(artifact, f)
        logger.info("Saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "PairwiseRecommender":
        """Load fitted recommender from directory (reads recommender.pkl)."""
        path = Path(path)
        with open(path / "recommender.pkl", "rb") as f:
            artifact = pickle.load(f)

        obj = cls(
            classifier=None,  # template not needed after loading
            feature_indices=artifact["feature_indices"],
            tied_threshold=artifact["tied_threshold"],
            feature_names=artifact.get("feature_names"),
            task_type=artifact.get("task_type"),
            metric_name=artifact.get("metric_name"),
        )
        obj.classifiers_ = artifact["classifiers"]
        obj.circuits_ = artifact["circuits"]
        obj.pairs_ = artifact["pairs"]
        obj.is_fitted_ = True
        return obj

    def _select_features(self, X: np.ndarray) -> np.ndarray:
        if self.feature_indices is not None:
            return X[:, self.feature_indices]
        return X