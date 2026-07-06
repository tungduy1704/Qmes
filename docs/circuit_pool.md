# Circuit Pool

Qmes ships a fixed pool of seven encoding circuits, spanning amplitude
embeddings, separable single-qubit rotations, and entangling circuits. All
are implemented in the bundled **Qsun** simulator and exposed through
[`CIRCUIT_POOL`](api/circuits.md).

| Name | Encoding | Entanglement | Input range |
|---|---|---|---|
| `unit` | Square-root amplitude (per-qubit) | None | $[0, 1]$ |
| `SRx` | Separable RX | None | $[0, \pi]$ |
| `RY` | Angle encoding (RY) | None | $[0, \pi]$ |
| `HERx` | Hardware-efficient RX | Linear (CX) | $[0, \pi]$ |
| `RY_CX` | Angle (RY) + linear CX | Linear (CX) | $[0, \pi]$ |
| `ZFM` | Z feature map | None | $[0, \pi]$ |
| `HD` | High-dimensional (RZ–RY–RZ) | Brickwork (SISWAP) | $[0, \pi]$ |

Most circuits map one feature to one qubit; `HD` encodes three rotation
angles per qubit. Datasets with more than 4 features are projected to
4 PCA components by the evaluators, matching the qubit budget of the
simulator.

## Input ranges

The evaluators scale features per circuit: to $[0, 1]$ if the circuit is in
[`UNIT_RANGE_CIRCUITS`](api/circuits.md), otherwise to $[0, \pi]$. Only
`unit` needs $[0, 1]$ - its amplitude encoding computes $\sqrt{x}$ and
$\sqrt{1-x}$, undefined outside that interval.

!!! warning
    If you register a custom circuit with the wrong range, nothing raises —
    the kernel silently collapses toward "everything looks similar". See
    [the UNIT_RANGE_CIRCUITS trap](advanced_usage.md#2-custom-circuit-evaluator)
    for a measured demonstration.

## Kernel

Every circuit induces a quantum fidelity kernel
$K(\boldsymbol{x}, \boldsymbol{x}') = |\langle\phi(\boldsymbol{x})|\phi(\boldsymbol{x}')\rangle|^2$,
computed by [`compute_kernel_matrix`](api/circuits.md). Building this
matrix costs $\mathcal{O}(n^2)$ circuit-state overlaps per dataset per
circuit - the cost Qmes's recommender avoids at inference time.

## Extending the pool

Adding a circuit is a dict insertion:

```python
import Qmes.circuits.registry as registry
registry.CIRCUIT_POOL["my_circuit"] = my_encode_fn
```

To ship it in a recommendation model, evaluate it on your benchmark
datasets with an [Evaluator](api/evaluators.md) and refit the
[Recommender](api/recommender.md) on the augmented meta-dataset — see
[Advanced Usage](advanced_usage.md#3-retrain-a-pairwiserecommender).
