# Quick Start

## Installation

```bash
pip install Qmes
```

## Classification example

```python
from sklearn.datasets import load_iris
from Qmes import get_extractor, load_default_recommender, recommend

# Load your dataset
X, y = load_iris(return_X_y=True)
# Binary classification only: keep 2 classes
mask = y < 2
X, y = X[mask], y[mask]

# Load pre-trained recommender (no quantum evaluation needed)
extractor = get_extractor("classification")
recommender = load_default_recommender("classification")

# Get circuit recommendation
result = recommend(X, y, extractor=extractor, recommender=recommender)

print("Top circuits:", result["top_k"]) # ['unit', 'RY', 'HERx']
print("Full ranking:", result["ranking"]) #  ['unit', 'RY', 'HERx', 'SRx', 'RY_CX', 'HD', 'ZFM']
```

## Regression example

```python
from sklearn.datasets import load_diabetes
from Qmes import get_extractor, load_default_recommender, recommend

X, y = load_diabetes(return_X_y=True)

extractor = get_extractor("regression")
recommender = load_default_recommender("regression")

result = recommend(X, y, extractor=extractor, recommender=recommender)

print("Top circuits:", result["top_k"]) # ['RY', 'unit', 'HERx']
print("Full ranking:", result["ranking"]) # ['RY', 'unit', 'HERx', 'HD', 'RY_CX', 'SRx', 'ZFM']
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

## Citation

If you use Qmes, please also cite Qsun, the quantum simulator bundled with this package:

```bibtex
@article{Nguyen_2022,
doi = {10.1088/2632-2153/ac5997},
url = {https://doi.org/10.1088/2632-2153/ac5997},
year = {2022},
month = {mar},
publisher = {IOP Publishing},
volume = {3},
number = {1},
pages = {015034},
author = {Nguyen, Quoc Chuong and Ho, Le Bin and Nguyen Tran, Lan and Nguyen, Hung Q},
title = {Qsun: an open-source platform towards practical quantum machine learning applications},
journal = {Machine Learning: Science and Technology},
abstract = {Currently, quantum hardware is restrained by noises and qubit numbers. Thus, a quantum virtual machine (QVM) that simulates operations of a quantum computer on classical computers is a vital tool for developing and testing quantum algorithms before deploying them on real quantum computers. Various variational quantum algorithms (VQAs) have been proposed and tested on QVMs to surpass the limitations of quantum hardware. Our goal is to exploit further the VQAs towards practical applications of quantum machine learning (QML) using state-of-the-art quantum computers. In this paper, we first introduce a QVM named Qsun, whose operation is underlined by quantum state wavefunctions. The platform provides native tools supporting VQAs. Especially using the parameter-shift rule, we implement quantum differentiable programming essential for gradient-based optimization. We then report two tests representative of QML: quantum linear regression and quantum neural network.}
}
```