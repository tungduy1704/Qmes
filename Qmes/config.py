"""Qmes/config.py

Single source of truth for constants shared across the train/inference
pipeline. These values are load-bearing for methodology, not just
convenience defaults:

- MAX_SAMPLES caps per-dataset subsampling. It must match between the
  meta-dataset construction (data/{clf,reg}/train.py, data/{clf,reg}/
  inference.py) and runner.preprocess_new_dataset() used at inference
  time -- a mismatch here previously caused divergent Oracle results
  (see data/{clf,reg}/train.py module docstring for the replace=True
  subsampling bug this is tied to).
- TIED_THRESHOLD defines what counts as a "tied" best circuit (absolute
  metric delta). It must match between meta-label construction at
  training time (recommender/selection.py::run_loo_evaluation) and
  scoring at inference time (inference/runner.py::evaluate_recommendation,
  recommender/pairwise.py::PairwiseRecommender) -- a mismatch silently
  changes what "correct" means between train and test.

Import these from here rather than redefining them locally.
"""

MAX_SAMPLES = 600
TIED_THRESHOLD = 0.01