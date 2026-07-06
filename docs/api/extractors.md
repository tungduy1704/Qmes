# Extractors

An extractor turns a dataset `(X, y)` into a fixed-length meta-feature
vector. Two concrete extractors ship with Qmes, both backed by
[problexity](https://problexity.readthedocs.io/en/latest/) complexity
measures — see [Meta-features](../meta_features.md) for the full list.

To write your own extractor, implement the three-member contract of
`BaseExtractor` below (`task_type`, `_feature_names`, `_extract_raw`);
a worked example is in [Advanced Usage](../advanced_usage.md#1-custom-extractor).

::: Qmes.get_extractor

::: Qmes.extractors.BaseExtractor
    options:
      filters: ["!^__"]

::: Qmes.extractors.ExtractionResult

::: Qmes.ClassificationExtractor

::: Qmes.RegressionExtractor
