import numpy as np
from Qmes.Qsun.Qencodes import ZFeatureMap_encode
from Qmes.Qsun.Qkernels import state_product

circuit_fn = lambda x: ZFeatureMap_encode(x)

X = np.array([
    [0.1, 0.5],
    [0.8, 0.3],
    [0.4, 0.9],
    [0.2, 0.7],
])

# ── Bản cũ ───────────────────────────────────────────────────────────────────
n = len(X)
K_old = np.zeros((n, n))
for i in range(n):
    phi_i = circuit_fn(X[i])
    for j in range(n):
        phi_j = circuit_fn(X[j])
        K_old[i, j] = state_product(phi_i, phi_j) ** 2

# ── Bản mới ──────────────────────────────────────────────────────────────────
states = [circuit_fn(x) for x in X]
K_new = np.zeros((n, n))
np.fill_diagonal(K_new, 1.0)
for i in range(n):
    for j in range(i + 1, n):
        v = state_product(states[i], states[j]) ** 2
        K_new[i, j] = K_new[j, i] = v

# ── So sánh ──────────────────────────────────────────────────────────────────
print("K_old:")
print(np.round(K_old, 6))
print("\nK_new:")
print(np.round(K_new, 6))
print("\nMax diff:", np.max(np.abs(K_old - K_new)))