"""
Filename: variational_test.py
Author: Santiago Nunez-Corrales
Date: 2026-05-12
Version: 1.0
Description:
    Tests for variational ansatz primitives: hw_eff_ansatz and qaoa_step.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import numpy as np
import pytest
from lccfq_lang.arch.isa import ISA
from lccfq_lang.lang.variational import hw_eff_ansatz, qaoa_step

from ._sim import (
    simulate,
    basis_state,
    hamiltonian_op,
    unitary_of_evolution,
)


@pytest.fixture
def isa():
    return ISA("test")


def symbols(instrs):
    return [i.symbol for i in instrs]


# ===========================================================================
# hw_eff_ansatz — structure and param binding
# ===========================================================================

class TestHwEffStructure:
    def test_default_l1_y_linear(self, isa):
        # n=3, L=1, rotations="y", entangler="linear"
        # Expected: 3 ry + 2 cx + 3 ry = 8 instrs
        n = 3
        params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        result = hw_eff_ansatz(isa, list(range(n)), params=params)
        assert symbols(result) == ["ry", "ry", "ry", "cx", "cx", "ry", "ry", "ry"]

    def test_layers_zero_is_just_initial_rotation(self, isa):
        n = 2
        params = [0.1, 0.2]
        result = hw_eff_ansatz(isa, list(range(n)), params=params, layers=0)
        assert symbols(result) == ["ry", "ry"]

    def test_two_axis_rotations(self, isa):
        # rotations="yz" → 2 rotations per qubit per layer
        n = 2
        # (L+1) * n * |rotations| = 2 * 2 * 2 = 8 params
        params = list(range(8))
        result = hw_eff_ansatz(
            isa, list(range(n)), params=params,
            layers=1, rotations="yz",
        )
        # initial: ry, rz, ry, rz; entangle: cx; final: ry, rz, ry, rz
        assert symbols(result) == [
            "ry", "rz", "ry", "rz",
            "cx",
            "ry", "rz", "ry", "rz",
        ]

    def test_param_binding_order(self, isa):
        # Verify params bind in (layer, qubit_position, rotation) order.
        n = 2
        params = [10.0, 20.0, 30.0, 40.0]  # L=1, "y", n=2 → 4 params
        result = hw_eff_ansatz(isa, [5, 7], params=params, layers=1)
        rys = [i for i in result if i.symbol == "ry"]
        # Initial layer: ry on q5 (10.0), ry on q7 (20.0)
        # Final layer (after cx): ry on q5 (30.0), ry on q7 (40.0)
        assert rys[0].target_qubits == [5] and rys[0].params == [10.0]
        assert rys[1].target_qubits == [7] and rys[1].params == [20.0]
        assert rys[2].target_qubits == [5] and rys[2].params == [30.0]
        assert rys[3].target_qubits == [7] and rys[3].params == [40.0]

    def test_three_axis_rotations_full_count(self, isa):
        # rotations="xyz", n=4, L=2 → 3*(2+1)*4 = 36 params
        n, L = 4, 2
        rots = "xyz"
        params = list(range((L + 1) * n * len(rots)))
        result = hw_eff_ansatz(
            isa, list(range(n)), params=params, layers=L, rotations=rots,
        )
        rxs = sum(1 for i in result if i.symbol == "rx")
        rys = sum(1 for i in result if i.symbol == "ry")
        rzs = sum(1 for i in result if i.symbol == "rz")
        assert rxs == rys == rzs == (L + 1) * n


class TestHwEffEntanglerChoice:
    def test_ring_topology(self, isa):
        n = 3
        params = [0.0] * (2 * n)  # L=1, "y"
        result = hw_eff_ansatz(
            isa, list(range(n)), params=params, layers=1, entangler="ring",
        )
        # ring on n=3 emits 3 CX
        cxs = [i for i in result if i.symbol == "cx"]
        assert len(cxs) == 3

    def test_cz_entangler(self, isa):
        n = 3
        params = [0.0] * (2 * n)
        result = hw_eff_ansatz(
            isa, list(range(n)), params=params, layers=1,
            entangle_gate="cz",
        )
        assert any(i.symbol == "cz" for i in result)
        assert not any(i.symbol == "cx" for i in result)


class TestHwEffSingleQubit:
    def test_n1_no_entangler(self, isa):
        # With n=1 there is nothing to entangle, so the ansatz reduces to
        # just rotations: (L+1) rotations per axis.
        params = [0.1, 0.2, 0.3]  # L=2, "y", n=1
        result = hw_eff_ansatz(isa, [0], params=params, layers=2)
        assert symbols(result) == ["ry", "ry", "ry"]
        assert not any(i.symbol == "cx" for i in result)


class TestHwEffValidation:
    def test_empty_target(self, isa):
        with pytest.raises(ValueError, match="at least 1"):
            hw_eff_ansatz(isa, [], params=[])

    def test_negative_layers(self, isa):
        with pytest.raises(ValueError, match="layers"):
            hw_eff_ansatz(isa, [0, 1], params=[0.0] * 2, layers=-1)

    def test_empty_rotations(self, isa):
        with pytest.raises(ValueError, match="rotations spec"):
            hw_eff_ansatz(isa, [0, 1], params=[], rotations="")

    def test_invalid_rotation_axis(self, isa):
        with pytest.raises(ValueError, match="rotations must contain"):
            hw_eff_ansatz(isa, [0, 1], params=[0.0] * 4, rotations="yw")

    def test_wrong_param_count(self, isa):
        with pytest.raises(ValueError, match="params length"):
            hw_eff_ansatz(isa, [0, 1], params=[0.0, 0.0, 0.0])  # expects 4

    def test_extra_params(self, isa):
        with pytest.raises(ValueError, match="params length"):
            hw_eff_ansatz(isa, [0, 1], params=[0.0] * 10)


# ===========================================================================
# qaoa_step — structure, default mixer, semantic equivalence
# ===========================================================================

def _circuit_unitary(circuit, n: int) -> np.ndarray:
    dim = 1 << n
    U = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        U[:, col] = simulate(circuit, n, basis_state(col, n))
    return U


class TestQaoaStepStructure:
    def test_cost_then_mixer(self, isa):
        # Cost = Z0Z1, Mixer = X0 + X1 (default).
        cost = [(1.0, {0: "Z", 1: "Z"})]
        result = qaoa_step(
            isa, [0, 1], gamma=0.7, beta=0.3, cost=cost,
        )
        syms = symbols(result)
        # Cost segment: cx, rz, cx (Z0Z1 evolution).
        # Mixer segment: h, rz, h, h, rz, h (X0 then X1).
        assert syms == ["cx", "rz", "cx", "h", "rz", "h", "h", "rz", "h"]

    def test_default_mixer_is_x_field(self, isa):
        # Empty cost → only the default mixer evolution remains.
        result = qaoa_step(
            isa, [0, 1, 2], gamma=0.0, beta=0.5, cost=[],
        )
        # 3 Pauli-X terms, each emits h-rz-h
        assert symbols(result) == ["h", "rz", "h"] * 3

    def test_custom_mixer(self, isa):
        # Custom mixer: just Y on qubit 0.
        cost = []
        mixer = [(1.0, {0: "Y"})]
        result = qaoa_step(
            isa, [0, 1], gamma=0.0, beta=0.4, cost=cost, mixer=mixer,
        )
        assert symbols(result) == ["rx", "rz", "rx"]


class TestQaoaStepSemantics:
    def test_diagonal_cost_exact(self, isa):
        # Cost has only Z terms (commuting), so cost evolution is exact.
        # Mixer = X-field (commuting), also exact. So qaoa_step at any
        # gamma/beta exactly equals exp(-i beta H_M) exp(-i gamma H_C).
        cost = [(1.0, {0: "Z", 1: "Z"}), (0.4, {0: "Z"})]
        n = 2
        gamma, beta = 0.7, 0.55
        circuit = qaoa_step(
            isa, list(range(n)), gamma=gamma, beta=beta, cost=cost,
        )
        Hc = hamiltonian_op(cost, n)
        Hm = hamiltonian_op([(1.0, {p: "X"}) for p in range(n)], n)
        expected = unitary_of_evolution(Hm, beta) @ unitary_of_evolution(
            Hc, gamma
        )
        U = _circuit_unitary(circuit, n)
        assert np.allclose(U, expected, atol=1e-9)

    def test_maxcut_two_qubit_amplitude(self, isa):
        # Apply qaoa_step starting from |+>|+> with cost = Z0Z1.
        # Result amplitudes should be e^{-i gamma} on |00>/|11> and
        # e^{+i gamma} on |01>/|10>, multiplied by the mixer rotation.
        cost = [(1.0, {0: "Z", 1: "Z"})]
        gamma, beta = 0.6, 0.0  # turn mixer off to inspect cost only
        n = 2
        # initial state |+>|+>
        psi0 = np.full(1 << n, 1.0 / np.sqrt(1 << n), dtype=complex)
        circuit = qaoa_step(
            isa, [0, 1], gamma=gamma, beta=beta, cost=cost,
        )
        psi1 = simulate(circuit, n, psi0)
        # eigenvalues of Z0Z1: |00>:+1, |01>:-1, |10>:-1, |11>:+1.
        # exp(-i*gamma*Z0Z1) on |+>|+> = sum_x exp(-i gamma E_x) |x> / 2.
        E = np.array([1, -1, -1, 1])
        expected = np.exp(-1j * gamma * E) / np.sqrt(1 << n)
        assert np.allclose(psi1, expected, atol=1e-9)


class TestQaoaStepValidation:
    def test_empty_target(self, isa):
        with pytest.raises(ValueError, match="at least 1"):
            qaoa_step(isa, [], gamma=0.1, beta=0.1, cost=[])

    def test_zero_gamma_and_beta_is_empty(self, isa):
        # Default mixer is non-empty but beta=0 → no mixer gates.
        # Cost is empty → no cost gates. Total: empty circuit.
        assert qaoa_step(
            isa, [0, 1], gamma=0.0, beta=0.0, cost=[]
        ) == []
