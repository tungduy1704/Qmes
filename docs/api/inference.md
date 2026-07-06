# Inference

End-to-end entry points. `recommend` is the main user-facing function:
extract meta-features, query the recommender, return a ranked circuit list -
no quantum evaluation at inference time. `evaluate_recommendation` optionally
checks recommendations against Oracle ground truth (this *does* run quantum
kernel evaluation).

::: Qmes.recommend

::: Qmes.preprocess_new_dataset

::: Qmes.evaluate_recommendation
