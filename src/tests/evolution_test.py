"""
Filename: evolution_test.py
Author: Santiago Nunez-Corrales
Date: 2026-05-12
Version: 1.0
Description:
    Tests for Hamiltonian evolution primitives: time_evolution and trotter_steps.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import numpy as np
import pytest
from lccfq_lang.arch.isa import ISA
from lccfq_lang.lang.evolution import time_evolution, trotter_steps

from ._sim import (
    simulate,
    basis_state,
    hamiltonian_op,
    unitary_of_evolution,
)


@pytest.fixture
def isa():
    return ISA("test")


def _circuit_unitary(circuit, n: int) -> np.ndarray:
    """Build the unitary the circuit implements by applying it to each
    computational-basis input."""
    dim = 1 << n
    U = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        psi = simulate(circuit, n, basis_state(col, n))
        U[:, col] = psi
    return U


# ===========================================================================
# time_evolution — single Pauli term exactness
# ===========================================================================

class TestSinglePauliExact:
    """For a single Pauli string the first-order Trotter step is exact."""

    @pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
    @pytest.mark.parametrize("t", [0.0, 0.3, 1.0, 2.5])
    def test_single_qubit_pauli(self, isa, pauli, t):
        c = 0.7
        ham = [(c, {0: pauli})]
        circuit = time_evolution(isa, [0], hamiltonian=ham, time=t)
        U = _circuit_unitary(circuit, 1)
        H = hamiltonian_op(ham, 1)
        expected = unitary_of_evolution(H, t)
        # exp(-iHt) up to a global phase. The Rz / Rx-based recipe
        # introduces no global phase so equality should be exact (modulo
        # numerical noise).
        assert np.allclose(U, expected, atol=1e-9)

    @pytest.mark.parametrize("paulis,n", [
        ({0: "Z", 1: "Z"}, 2),
        ({0: "X", 1: "X"}, 2),
        ({0: "Y", 1: "Y"}, 2),
        ({0: "X", 1: "Z"}, 2),
        ({0: "Z", 1: "Y", 2: "X"}, 3),
        ({0: "Y", 2: "Z"}, 3),
    ])
    @pytest.mark.parametrize("t", [0.2, 1.1])
    def test_multi_qubit_pauli_string(self, isa, paulis, n, t):
        c = 0.4
        ham = [(c, paulis)]
        circuit = time_evolution(isa, list(range(n)),
                                 hamiltonian=ham, time=t)
        U = _circuit_unitary(circuit, n)
        H = hamiltonian_op(ham, n)
        expected = unitary_of_evolution(H, t)
        assert np.allclose(U, expected, atol=1e-9), (
            f"mismatch for paulis={paulis} n={n} t={t}"
        )


# ===========================================================================
# time_evolution — commuting terms are exact, non-commuting are first-order
# ===========================================================================

class TestCommutingTerms:
    """A Hamiltonian of mutually commuting Pauli strings: one Trotter step is exact."""

    def test_two_commuting_z_strings(self, isa):
        ham = [
            (0.5, {0: "Z"}),
            (0.3, {0: "Z", 1: "Z"}),
        ]
        t = 0.7
        circuit = time_evolution(isa, [0, 1], hamiltonian=ham, time=t)
        U = _circuit_unitary(circuit, 2)
        expected = unitary_of_evolution(hamiltonian_op(ham, 2), t)
        assert np.allclose(U, expected, atol=1e-9)


class TestNonCommutingFirstOrder:
    """For small t the first-order Trotter step matches exp(-iHt) up to O(t**2)."""

    def test_error_scales_with_t_squared(self, isa):
        # Transverse-field Ising on 2 qubits: ZZ + X+X. Terms don't commute.
        ham = [
            (1.0, {0: "Z", 1: "Z"}),
            (0.5, {0: "X"}),
            (0.5, {1: "X"}),
        ]
        target = [0, 1]
        H = hamiltonian_op(ham, 2)

        errors = []
        for t in (0.4, 0.2, 0.1, 0.05):
            U = _circuit_unitary(
                time_evolution(isa, target, hamiltonian=ham, time=t), 2
            )
            expected = unitary_of_evolution(H, t)
            errors.append(np.linalg.norm(U - expected))

        # Halving t should shrink the error by ~4x for first-order Trotter.
        # Check the last-step ratio with generous slack.
        ratio = errors[-2] / errors[-1]
        assert 3.0 < ratio < 5.5, (
            f"expected ~4x error reduction, got ratio={ratio} "
            f"with errors={errors}"
        )


# ===========================================================================
# time_evolution — degenerate / identity / zero
# ===========================================================================

class TestDegenerateTerms:
    def test_zero_time_is_identity(self, isa):
        ham = [(1.0, {0: "X"}), (0.5, {1: "Z"})]
        circuit = time_evolution(isa, [0, 1], hamiltonian=ham, time=0.0)
        assert circuit == []

    def test_zero_coefficient_skipped(self, isa):
        ham = [(0.0, {0: "X"}), (1.0, {1: "Z"})]
        circuit = time_evolution(isa, [0, 1], hamiltonian=ham, time=0.5)
        # Only the Z term survives → one rz, no h or rx
        symbols = [i.symbol for i in circuit]
        assert "h" not in symbols
        assert "rx" not in symbols
        assert symbols.count("rz") == 1

    def test_identity_paulis_skipped(self, isa):
        ham = [(1.0, {0: "I", 1: "I"})]
        circuit = time_evolution(isa, [0, 1], hamiltonian=ham, time=1.0)
        assert circuit == []

    def test_empty_hamiltonian(self, isa):
        circuit = time_evolution(isa, [0, 1], hamiltonian=[], time=1.0)
        assert circuit == []


# ===========================================================================
# time_evolution — structural checks
# ===========================================================================

class TestStructure:
    def test_z_only_term_uses_only_cx_and_rz(self, isa):
        ham = [(1.0, {0: "Z", 1: "Z", 2: "Z"})]
        circuit = time_evolution(isa, [0, 1, 2], hamiltonian=ham, time=0.7)
        symbols = {i.symbol for i in circuit}
        assert symbols == {"cx", "rz"}

    def test_x_term_uses_h_around_rz(self, isa):
        ham = [(1.0, {0: "X"})]
        circuit = time_evolution(isa, [0], hamiltonian=ham, time=0.7)
        symbols = [i.symbol for i in circuit]
        assert symbols == ["h", "rz", "h"]

    def test_y_term_uses_rx_pi_over_two(self, isa):
        ham = [(1.0, {0: "Y"})]
        circuit = time_evolution(isa, [0], hamiltonian=ham, time=0.7)
        symbols = [i.symbol for i in circuit]
        assert symbols == ["rx", "rz", "rx"]
        # The bracketing Rx angles must be +pi/2 and -pi/2
        assert np.isclose(circuit[0].params[0], np.pi / 2)
        assert np.isclose(circuit[2].params[0], -np.pi / 2)


# ===========================================================================
# time_evolution — validation
# ===========================================================================

class TestValidation:
    def test_empty_target(self, isa):
        with pytest.raises(ValueError, match="at least 1"):
            time_evolution(isa, [], hamiltonian=[], time=1.0)

    def test_pauli_out_of_range(self, isa):
        with pytest.raises(ValueError, match="not in"):
            time_evolution(
                isa, [0, 1], hamiltonian=[(1.0, {5: "X"})], time=1.0
            )

    def test_invalid_pauli_char(self, isa):
        with pytest.raises(ValueError, match="must be one of"):
            time_evolution(
                isa, [0], hamiltonian=[(1.0, {0: "Q"})], time=1.0
            )

    def test_invalid_pauli_type(self, isa):
        with pytest.raises(TypeError):
            time_evolution(
                isa, [0], hamiltonian=[(1.0, {0: 1})], time=1.0
            )


# ===========================================================================
# trotter_steps — equivalence to time_evolution and convergence
# ===========================================================================

class TestTrotterStepsEquivalence:
    def test_single_step_first_order_equals_time_evolution(self, isa):
        ham = [(0.7, {0: "Z", 1: "X"}), (0.3, {1: "Y"})]
        t = 0.6
        a = _circuit_unitary(
            trotter_steps(isa, [0, 1], hamiltonian=ham, time=t, steps=1), 2
        )
        b = _circuit_unitary(
            time_evolution(isa, [0, 1], hamiltonian=ham, time=t), 2
        )
        assert np.allclose(a, b, atol=1e-9)

    def test_commuting_terms_exact_any_steps(self, isa):
        # Commuting Hamiltonian: any number of steps gives exact evolution.
        ham = [(0.4, {0: "Z"}), (1.1, {0: "Z", 1: "Z"})]
        t = 1.3
        expected = unitary_of_evolution(hamiltonian_op(ham, 2), t)
        for r in (1, 3, 7):
            U = _circuit_unitary(
                trotter_steps(isa, [0, 1], hamiltonian=ham, time=t, steps=r),
                2,
            )
            assert np.allclose(U, expected, atol=1e-9)


class TestTrotterStepsConvergence:
    """For a non-commuting Hamiltonian, first-order error scales like
    O(t**2 / steps); second-order like O(t**3 / steps**2)."""

    HAM = [
        (1.0, {0: "Z", 1: "Z"}),
        (0.7, {0: "X"}),
        (0.5, {1: "X"}),
    ]
    T = 0.8

    def _error(self, isa, steps, order):
        circuit = trotter_steps(
            isa, [0, 1], hamiltonian=self.HAM, time=self.T,
            steps=steps, order=order,
        )
        U = _circuit_unitary(circuit, 2)
        exact = unitary_of_evolution(hamiltonian_op(self.HAM, 2), self.T)
        return np.linalg.norm(U - exact)

    def test_first_order_scales_inversely_with_steps(self, isa):
        e1 = self._error(isa, steps=4, order=1)
        e2 = self._error(isa, steps=8, order=1)
        # Doubling steps should ~halve first-order error.
        ratio = e1 / e2
        assert 1.6 < ratio < 2.4, (
            f"first-order: expected ~2x ratio, got {ratio} "
            f"({e1=}, {e2=})"
        )

    def test_second_order_scales_inversely_with_steps_squared(self, isa):
        e1 = self._error(isa, steps=4, order=2)
        e2 = self._error(isa, steps=8, order=2)
        # Doubling steps should ~quarter second-order error.
        ratio = e1 / e2
        assert 3.4 < ratio < 5.0, (
            f"second-order: expected ~4x ratio, got {ratio} "
            f"({e1=}, {e2=})"
        )

    def test_second_order_more_accurate_than_first(self, isa):
        e1 = self._error(isa, steps=8, order=1)
        e2 = self._error(isa, steps=8, order=2)
        assert e2 < e1


class TestTrotterStepsDegenerate:
    def test_zero_time(self, isa):
        ham = [(1.0, {0: "X"})]
        assert trotter_steps(
            isa, [0], hamiltonian=ham, time=0.0, steps=4
        ) == []

    def test_empty_hamiltonian(self, isa):
        assert trotter_steps(
            isa, [0], hamiltonian=[], time=1.0, steps=4
        ) == []


class TestTrotterStepsValidation:
    def test_empty_target(self, isa):
        with pytest.raises(ValueError, match="at least 1"):
            trotter_steps(isa, [], hamiltonian=[], time=1.0, steps=1)

    def test_zero_steps(self, isa):
        with pytest.raises(ValueError, match="steps"):
            trotter_steps(
                isa, [0], hamiltonian=[(1.0, {0: "X"})], time=1.0, steps=0
            )

    def test_bad_order(self, isa):
        with pytest.raises(ValueError, match="order"):
            trotter_steps(
                isa, [0], hamiltonian=[(1.0, {0: "X"})],
                time=1.0, steps=1, order=3,
            )
