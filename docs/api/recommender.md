# Recommender

The recommender is a classical meta-learner: given a dataset's meta-feature
vector, it predicts a ranking over the circuit pool. Most users only need
`load_default_recommender`, which returns the pre-trained bundle shipped
with the package. `get_recommender` builds a fresh (unfitted) recommender
for retraining; the model-selection utilities below drive the offline
search over classifiers and feature subsets.

::: Qmes.load_default_recommender

::: Qmes.get_recommender

::: Qmes.PairwiseRecommender

## Model selection utilities

::: Qmes.run_loo_evaluation

::: Qmes.recommender.select_features_mi

::: Qmes.recommender.DEFAULT_CLASSIFIERS
