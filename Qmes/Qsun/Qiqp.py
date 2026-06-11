import numpy as np
import itertools
import cmath as cm
from Qmes.Qsun.Qgates import H, RZ, CNOT
from Qmes.Qsun.Qcircuit import Qubit


class IQP:
    """
    Class representing an Instantaneous Quantum Polynomial (IQP) circuit.
    """

    def __init__(self, n_qubit=3, layers=2, theta=0.7):
        self.n_qubit = n_qubit
        self.layers = layers
        self.theta = theta

    def MultiRZ(self, wavefunction, wires, theta):
        states = wavefunction.state
        amplitude = wavefunction.amplitude
        qubit_num = len(states[0])
        new_amp = np.zeros(2 ** qubit_num, dtype=complex)

        if not wires:
            raise ValueError("MultiRZ: At least one wire must be provided")
        if len(wires) != len(set(wires)):
            raise ValueError("MultiRZ: All wires must be distinct")
        for w in wires:
            if w >= qubit_num or w < 0:
                raise TypeError(f"Index {w} out of range [0, {qubit_num-1}]")

        for i in range(2 ** qubit_num):
            parity = sum(states[i][wire_idx] == '1' for wire_idx in wires)
            eigenval = (-1) ** parity
            phase = cm.exp(-0.5j * theta * eigenval)
            new_amp[i] = amplitude[i] * phase

        wavefunction.amplitude = new_amp

        if len(wires) == 2:
            i, j = int(wires[0]), int(wires[1])
            wavefunction.visual.append([i, j, 'CP', '0'])
        else:
            base = int(wires[0])
            for w in wires[1:]:
                wavefunction.visual.append([base, int(w), 'CP', '0'])

    def MultiRZ_decompose(self, wavefunction, wires, theta):
        if not wires:
            raise ValueError("MultiRZ: At least one wire must be provided")
        qubit_num = len(wavefunction.state[0])
        if len(wires) != len(set(wires)):
            raise ValueError("MultiRZ: All wires must be distinct")
        for w in wires:
            if w >= qubit_num or w < 0:
                raise TypeError(f"Index {w} is out of range [0, {qubit_num-1}]")

        for i in range(len(wires) - 1, 0, -1):
            CNOT(wavefunction, wires[i], wires[i - 1])
        RZ(wavefunction, wires[0], theta)
        for i in range(1, len(wires)):
            CNOT(wavefunction, wires[i], wires[i - 1])

    def MRZ_pair(self, wavefunction, wires, theta):
        if len(wires) < 2:
            raise ValueError("Require at least 2 wires for MRZ_pair")
        for i, j in itertools.combinations(wires, 2):
            self.MultiRZ(wavefunction, [i, j], theta)

    def MRZ_chain(self, wavefunction, wires, theta):
        if len(wires) < 2:
            raise ValueError("Require at least 2 wires for MRZ_chain")
        for a, b in zip(wires, wires[1:]):
            self.MultiRZ(wavefunction, [a, b], theta)

    def circuit(self, features, use_pair=True):
        if len(features) != self.n_qubit:
            raise ValueError("Length of features must equal n_qubit")

        wf = Qubit(self.n_qubit)
        for _ in range(self.layers):
            for q in range(self.n_qubit):
                H(wf, q)
                RZ(wf, q, features[q])
            if use_pair:
                self.MRZ_pair(wf, list(range(self.n_qubit)), self.theta)
            else:
                self.MRZ_chain(wf, list(range(self.n_qubit)), self.theta)
        return wf

    def visualize(self, features):
        wf = self.circuit(features)
        wf.visual_circuit()
        return wf