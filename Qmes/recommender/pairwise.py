"""Qmes/recommender/pairwise.py

Pairwise OvO recommender: trains one binary comparator per circuit pair, aggregates votes to rank circuits.
"""
from __future__ import annotations

import logging
import importlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from Qmes.config import TIED_THRESHOLD

logger = logging.getLogger(__name__)


class PairwiseRecommender:
    """Pairwise one-vs-one (OvO) circuit recommender.

    Trains one binary comparator per circuit pair (21 comparators for the
    default 7-circuit pool), each an independent clone of the base
    ``classifier``. Each comparator learns, from a dataset's
    meta-features, which of its two circuits scores higher. At prediction
    time every pair casts one vote and circuits are ranked by total
    vote count.

    A single class serves both classification and regression: behavior is
    determined by the ``classifier`` template and the training data, not
    by ``task_type`` (which is stored for validation against the extractor
    at inference time).

    Typical usage:

    - ``load_default_recommender(task)`` — pre-trained bundle, ready to use.
    - ``get_recommender(task, clf, ...)`` then ``fit()`` — retrain your own.

    Attributes
    ----------
    circuits_ : list[str]
        Circuit names, set after ``fit()`` (pivot index order).
    pairs_ : list[tuple[str, str]]
        All circuit pairs, set after ``fit()``.
    classifiers_ : dict[tuple[str, str], object]
        Fitted comparator (a clone of the base ``classifier``) per pair,
        set after ``fit()``.
    is_fitted_ : bool
        True once ``fit()`` has run.
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
        self.classifier = classifier
        self.feature_indices = feature_indices
        self.tied_threshold = tied_threshold
        self.feature_names = feature_names
        self.task_type = task_type
        self.metric_name = metric_name

        # Set after fit()
        self.train_meta_ = None
        self.train_pivot_ = None
        self.circuits_: list[str] = []
        self.pairs_: list[tuple[str, str]] = []
        self.classifiers_: dict[tuple[str, str], object] = {}
        self.is_fitted_ = False

    def fit(
        self,
        meta_features: np.ndarray | pd.DataFrame,
        pivot_scores: pd.DataFrame,
    ) -> "PairwiseRecommender":
        """Train one comparator per circuit pair.

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

        self.train_meta_ = meta_features
        self.train_pivot_ = pivot_scores.copy()
            
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
            "Fitted %d pairwise comparators for %d circuits",
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
        """Save the recommender to a directory as a format-v2 bundle.

        Writes two files: ``recommender.npz`` (training meta-feature
        matrix and pivot score values) and ``recommender.json``
        (classifier spec and configuration). Fitted estimators are NOT
        persisted; load() refits from the stored training data instead,
        so the bundle carries no pickled sklearn objects and stays
        independent of the sklearn version that produced it.

        Parameters
        ----------
        path : str or Path
            Target directory. Created if it does not exist.

        Raises
        ------
        RuntimeError
            If called before fit().
        TypeError
            If the classifier's get_params() contains values that are
            not JSON-serializable (e.g. estimator objects).
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before save().")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path / "recommender.npz",
            train_meta=self.train_meta_,
            pivot_values=self.train_pivot_.values,
        )
        spec = {
            "class": f"{type(self.classifier).__module__}.{type(self.classifier).__qualname__}",
            "params": self.classifier.get_params(),
        }
        meta = {
            "format_version": 2,
            "task_type": self.task_type,
            "metric_name": self.metric_name,
            "feature_names": self.feature_names,
            "feature_indices": (
                None if self.feature_indices is None
                else [int(i) for i in self.feature_indices]
            ),
            "tied_threshold": self.tied_threshold,
            "circuits": list(self.train_pivot_.index),
            "datasets": list(self.train_pivot_.columns),
            "classifier_spec": spec,
        }
        with open(path / "recommender.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PairwiseRecommender":
        """Load a format-v2 bundle and refit from its stored training data.

        The classifier is reconstructed from the spec in
        ``recommender.json`` and refit on the training matrices in
        ``recommender.npz``. Refitting is cheap and deterministic for
        the bundled kNN (no random state), and removes any coupling to
        the sklearn version that created the bundle.

        Parameters
        ----------
        path : str or Path
            Directory containing ``recommender.npz`` and
            ``recommender.json``.

        Returns
        -------
        PairwiseRecommender
            A fitted recommender equivalent to the one that was saved.

        Raises
        ------
        ValueError
            If the directory contains a legacy format-v1 pickle bundle.
        FileNotFoundError
            If the bundle files are missing.
        """
        path = Path(path)
        if (path / "recommender.pkl").exists() and not (path / "recommender.json").exists():
            raise ValueError(
                f"{path} contains a format-v1 pickle bundle, which is no longer "
                f"supported. Re-run the training pipeline and save() again."
            )
        with open(path / "recommender.json") as f:
            meta = json.load(f)

        module, _, name = meta["classifier_spec"]["class"].rpartition(".")
        clf_cls = getattr(importlib.import_module(module), name)
        classifier = clf_cls(**meta["classifier_spec"]["params"])

        data = np.load(path / "recommender.npz")
        pivot = pd.DataFrame(
            data["pivot_values"], index=meta["circuits"], columns=meta["datasets"]
        )
        obj = cls(
            classifier=classifier,
            feature_indices=meta["feature_indices"],
            tied_threshold=meta["tied_threshold"],
            feature_names=meta["feature_names"],
            task_type=meta["task_type"],
            metric_name=meta["metric_name"],
        )
        return obj.fit(data["train_meta"], pivot)

    def _select_features(self, X: np.ndarray) -> np.ndarray:
        if self.feature_indices is not None:
            return X[:, self.feature_indices]
        return X