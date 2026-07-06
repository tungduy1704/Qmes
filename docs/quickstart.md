# Quick Start

## Installation

```bash
pip install git+https://github.com/tungduy1704/Qmes.git
```

or, for a development install:

```bash
git clone https://github.com/tungduy1704/Qmes.git
cd Qmes
pip install -e .
```

!!! note
    Rebuilding the meta-dataset from UCI sources (the offline training
    pipeline in `Qmes/data/`) requires the optional `data` extra:
    `pip install -e ".[data]"`. Inference and the default recommenders
    do not need it.

## Classification example

```python
from sklearn.datasets import load_breast_cancer
from Qmes import get_extractor, load_default_recommender, recommend

X, y = load_breast_cancer(return_X_y=True)

extractor = get_extractor("classification")
recommender = load_default_recommender("classification")
result = recommend(X, y, extractor=extractor, recommender=recommender)

print("Top circuits:", result["top_k"])
# Top circuits: ['unit', 'RY', 'HERx']
print("Full ranking:", result["ranking"])
# Full ranking: ['unit', 'RY', 'HERx', 'SRx', 'RY_CX', 'HD', 'ZFM']
print("Vote counts:", result['votes'])
# Vote counts: {'unit': 6, 'SRx': 3, 'RY': 5, 'HERx': 4, 'RY_CX': 2, 'ZFM': 0, 'HD': 1}
```

## Regression example

```python
from sklearn.datasets import load_diabetes
from Qmes import get_extractor, load_default_recommender, recommend

X, y = load_diabetes(return_X_y=True)

extractor = get_extractor("regression")
recommender = load_default_recommender("regression")

result = recommend(X, y, extractor=extractor, recommender=recommender)

print("Top circuits:", result["top_k"])
# Top circuits: ['RY', 'unit', 'HERx']
print("Full ranking:", result["ranking"])
# Full ranking: ['RY', 'unit', 'HERx', 'HD', 'RY_CX', 'SRx', 'ZFM']
print("Vote counts:", result['votes'])
# Vote counts: {'unit': 5, 'SRx': 1, 'RY': 6, 'HERx': 4, 'RY_CX': 2, 'ZFM': 0, 'HD': 3}
```

## Understanding the output

| Key | Type | Description |
|---|---|---|
| `ranking` | `list[str]` | All 7 circuits sorted by OvO votes |
| `top_k` | `list[str]` | First `top_k` elements of `ranking` (default: 3) |
| `votes` | `dict[str, int]` | Raw vote count per circuit |
| `meta_features` | `np.ndarray` | Complexity features extracted from your dataset |

## Available circuits

```python
from Qmes import get_circuit_names
print(get_circuit_names())
# ['unit', 'SRx', 'RY', 'HERx', 'RY_CX', 'ZFM', 'HD']
```