# API Reference

The API reflects the three-stage pipeline described in the paper: an
**Extractor** turns a dataset into a meta-feature vector, an **Evaluator**
(offline only) scores circuits to build ground-truth labels, and a
**Recommender** predicts a circuit ranking from meta-features. The
**Inference** entry points connect them together; the **Circuits** module
holds the encoding-circuit pool and kernel computation.

| Page | Contents |
|---|---|
| [Inference](inference.md) | `recommend`, `preprocess_new_dataset`, `evaluate_recommendation` - the end-to-end entry points |
| [Extractors](extractors.md) | `get_extractor`, `BaseExtractor`, `ExtractionResult`, task-specific extractors |
| [Evaluators](evaluators.md) | `get_evaluator`, `BaseEvaluator`, task-specific Oracles, `filter_degenerate_datasets` |
| [Recommender](recommender.md) | `load_default_recommender`, `get_recommender`, `PairwiseRecommender`, model-selection utilities |
| [Circuits](circuits.md) | `CIRCUIT_POOL`, `UNIT_RANGE_CIRCUITS`, kernel-matrix computation |

For a walkthrough, start with the [Quick Start](../quickstart.md); to extend
Qmes with your own extractor, circuit, or recommender, see
[Advanced Usage](../advanced_usage.md).
