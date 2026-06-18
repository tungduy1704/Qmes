"""Qmes/_models/__init__.py

Default pre-trained PairwiseRecommender bundles, shipped as package data
(see [tool.setuptools.package-data] in pyproject.toml).

These exist so `pip install Qmes` gives a user a working recommend() call
out of the box, without first running the train pipeline in scripts/.
Layout: _models/<task_type>/{recommender.pkl, config_meta.json}.
"""