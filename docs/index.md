# Qmes

<p style="font-size: 1.3rem; font-weight: 500;">Quantum Meta-learning for Encoding Selection</p>

Qmes recommends the most suitable quantum encoding circuit for a tabular dataset — without running quantum evaluation at inference time.

[![CI](https://github.com/tungduy1704/Qmes/actions/workflows/ci.yml/badge.svg)](https://github.com/tungduy1704/Qmes/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/tungduy1704/Qmes)

## The problem

Selecting a quantum encoding circuit for a new dataset currently requires 
exhaustive evaluation: running all candidate circuits on the dataset and 
comparing their performance. This is expensive — evaluating a single circuit 
requires building an $n \times n$ kernel matrix via quantum state simulations, 
and this must be repeated for every candidate.

## What Qmes does

Qmes solves this with meta-learning:

1. **Offline**: evaluate 7 candidate circuits on 105 classification datasets and 86 regression datasets, build a meta-dataset of (complexity features → best circuit)
2. **Online**: for a new dataset, extract complexity features in seconds, query a pre-trained recommender, get a ranked circuit list — no quantum evaluation needed

## Quick example

```python
from sklearn.datasets import load_iris
from Qmes import get_extractor, load_default_recommender, recommend

X, y = load_iris(return_X_y=True)
mask = y < 2
X, y = X[mask], y[mask]

extractor = get_extractor("classification")
recommender = load_default_recommender("classification")
result = recommend(X, y, extractor=extractor, recommender=recommender)

print(result["top_k"])    # ['unit', 'RY', 'HERx']
print(result["ranking"])  # ['unit', 'RY', 'HERx', 'SRx', 'RY_CX', 'HD', 'ZFM']
```

## Supported tasks

| Task | Metric | Meta-features |
|---|---|---|
| Tabular classification | MCC (kernel SVC) | 22-dim Problexity |
| Tabular regression | R² (kernel Ridge) | 12-dim Problexity |

## Quick install

```bash
pip install Qmes
```

## 7 candidate circuits

`unit`, `SRx`, `RY`, `HERx`, `RY_CX`, `ZFM`, `HD`

## Citation

If you use Qmes in your research, please cite:

```bibtex
@misc{tung2026automatedselectionquantumencoding,
      title={Towards Automated Selection of Quantum Encoding Circuits via Meta-Learning}, 
      author={Dao Duy Tung and Nguyen Quoc Chuong and Vu Tuan Hai and Le Bin Ho and Lan Nguyen Tran},
      year={2026},
      eprint={2604.19076},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2604.19076}, 
}
```

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