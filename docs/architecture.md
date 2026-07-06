# Architecture

Qmes implements a three-stage meta-learning pipeline. The quantum-expensive
work happens **once, offline**; a user at inference time runs only classical
code.

![Qmes workflow](assets/img/scheme.png)

*Stage (a): data processing and ground-truth label generation via the
Evaluator. Stage (b): recommender training over classifier and
feature-subset configurations. Stage (c): the fitted recommender produces a
ranked circuit list for a new dataset.*

## Core components

Three pluggable components, each an abstract base class with one concrete
implementation per task type:

| Component | Base class | Role | Phase |
|---|---|---|---|
| Extractor | [`BaseExtractor`](api/extractors.md) | Dataset → fixed-length complexity meta-feature vector | offline **and** inference |
| Evaluator (Oracle) | [`BaseEvaluator`](api/evaluators.md) | Score every circuit on a dataset via quantum-kernel → meta-labels | **offline only** |
| Recommender | [`PairwiseRecommender`](api/recommender.md) | Meta-features → ranked circuit list (one-vs-one votes) | offline fit, inference predict |

The split is the point: the Evaluator is the only component that runs
quantum simulation, and it never runs at inference time. `recommend()`
touches the Extractor and Recommender only.

## Utility modules

| Module | Contents |
|---|---|
| Preprocessing (`Qmes.data.preprocessing`) | Categorical encoding, median imputation, subsampling to 600 samples |
| Circuit registry ([`Qmes.circuits`](api/circuits.md)) | `CIRCUIT_POOL` (7 circuits), quantum kernel-matrix computation |
| Model selection ([`Qmes.recommender.selection`](api/recommender.md#model-selection-utilities)) | LOO search over 14 classifiers × MI-selected feature subsets |
| Inference runner ([`Qmes.inference`](api/inference.md)) | `recommend`, `preprocess_new_dataset`, `evaluate_recommendation` |

The bundled **Qsun** simulator provides the quantum backend for all circuit
evaluations and is invoked exclusively during the offline phase.

## Extending Qmes

Both base classes use a small fixed contract - a subclass implements three
members and the base class handles validation, sanitization, and the public
API:

- **New extractor**: `task_type`, `_feature_names`, `_extract_raw(X, y)`
- **New Oracle**: `task_type`, `metric_name`, `evaluate_circuit(X, y, name)`
- **New circuit**: one dict entry in `CIRCUIT_POOL` — no subclassing

Worked, executed examples for all three are in
[Advanced Usage](advanced_usage.md).
