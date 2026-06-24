# Meta-features

Qmes describes each dataset with complexity measures computed by
[problexity](https://problexity.readthedocs.io/en/latest/), an open-source
library implementing the data-complexity measures of Lorena et al. Qmes does
not define its own measures — it uses problexity's outputs directly as the
meta-feature vector passed to the recommender.

## Classification — 22 measures

| Category | Measures |
|---|---|
| [Feature-based](https://problexity.readthedocs.io/en/latest/feature_based_api.html) | f1, f1v, f2, f3, f4 |
| [Linearity](https://problexity.readthedocs.io/en/latest/linearity_api.html) | l1, l2, l3 |
| [Neighborhood](https://problexity.readthedocs.io/en/latest/neighborhood_api.html) | n1, n2, n3, n4 |
| [Network](https://problexity.readthedocs.io/en/latest/network_api.html) | t1, lsc, density, cls_coef, hubs |
| [Dimensionality](https://problexity.readthedocs.io/en/latest/dimensionality_api.html) | t2, t3, t4 |
| [Class imbalance](https://problexity.readthedocs.io/en/latest/class_imbalance_api.html) | c1, c2 |

## Regression — 12 measures

| Category | Measures |
|---|---|
| [Correlation](https://problexity.readthedocs.io/en/latest/correlation_api.html) | c1, c2, c3, c4 |
| [Linearity](https://problexity.readthedocs.io/en/latest/linearity_api.html) | l1, l2, l3 |
| [Smoothness](https://problexity.readthedocs.io/en/latest/smoothness_api.html) | s1, s2, s3, s4 |
| [Dimensionality](https://problexity.readthedocs.io/en/latest/dimensionality_api.html) | t2 |

See the [problexity documentation](https://problexity.readthedocs.io/en/latest/)
for the precise definition of each measure.

```bibtex
@article{komorniczak2023problexity,
title={problexity—An open-source Python library for supervised learning problem complexity assessment},
author={Komorniczak, Joanna and Ksieniewicz, Pawe{\l}},
journal={Neurocomputing},
volume={521},
pages={126--136},
year={2023},
publisher={Elsevier}
}

@article{lorena2018complex,
title={How complex is your classification problem},
author={Lorena, A and Garcia, L and Lehmann, Jens and Souto, M and Ho, T},
journal={A survey on measuring classification complexity. arXiv},
year={2018}
}
```