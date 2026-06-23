# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-06-23

### Added
- Tabular classification task: ClassificationExtractor (22-dim Problexity),
  ClassificationEvaluator (kernel SVC + MCC)
- Tabular regression task: RegressionExtractor (12-dim Problexity),
  RegressionEvaluator (kernel Ridge + R²)
- PairwiseRecommender: OvO kNN meta-learner, shared across task types
- Pre-trained default recommenders shipped in `Qmes/_models/`:
  kNN_top5 (classification, regret=0.0183), kNN_top10 (regression, regret=0.0150), 
- Public API: `recommend`, `load_default_recommender`, `get_extractor`,
  `get_evaluator`, `get_recommender`, `evaluate_recommendation`,
  `preprocess_new_dataset`, `run_loo_evaluation`, `filter_degenerate_datasets`,
  `PairwiseRecommender`, `CIRCUIT_POOL`, `get_circuit_names`
- 7 candidate circuits: unit, SRx, RY, HERx, RY_CX, ZFM, HD
- Pluggable architecture via abstract base classes: BaseExtractor, BaseEvaluator
- Bundled Qsun quantum simulator (Qsun v1.0, MIT License)
- Test suite: 48 tests across 5 files, CI passing on Python 3.10/3.11/3.12
- Documentation: mkdocs-material site at https://tungduy1704.github.io/Qmes/