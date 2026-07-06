# Evaluators

An evaluator (the *Oracle*) scores every circuit in the pool on a dataset
via cross-validated quantum-kernel methods. It is used **offline only** —
to build the meta-dataset labels and to validate recommendations. Inference
never touches it.

A new Oracle implements the three-member contract of `BaseEvaluator`
(`task_type`, `metric_name`, `evaluate_circuit`); the base class provides
`evaluate_all` and `build_pivot` on top.

::: Qmes.get_evaluator

::: Qmes.evaluators.BaseEvaluator

::: Qmes.ClassificationEvaluator

::: Qmes.RegressionEvaluator

::: Qmes.filter_degenerate_datasets
