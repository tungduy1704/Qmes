"""Qmes/_models/__init__.py

Default pre-trained PairwiseRecommender bundles, shipped as package data
(see [tool.setuptools.package-data] in pyproject.toml).

These exist so `pip install Qmes` gives a working
`load_default_recommender(task_type)` out of the box, without first
running the offline train pipeline in scripts/.

Layout: _models/<task_type>/
    recommender.npz   - training meta-feature matrix + pivot score values
    recommender.json  - classifier spec + configuration (format v2;
                        load() refits deterministically, no pickled sklearn)
    config_meta.json  - bundle provenance (config name, selected feature
                        names, LOO metrics) for reproducibility/inspection
"""