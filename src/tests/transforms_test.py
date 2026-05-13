"""
Filename: transforms_test.py
Author: Santiago Nunez-Corrales
Date: 2026-05-12
Version: 1.0
Description:
    Tests for unitary transform blocks: qft and iqft.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import math
import numpy as np
import pytest
from lccfq_lang.arch.isa import ISA
from lccfq_lang.lang.transforms import qft, iqft

from ._sim import simulate, basis_state


@pytest.fixture
def isa():
    return ISA("test")


def symbols(instrs):
    return [i.symbol for i in instrs]


def count(instrs, symbol):
    return sum(1 for i in instrs if i.symbol == symbol)


# ===========================================================================
# qft — structural correctness
# ===========================================================================

class TestQFTSingleQubit:
    def test_single_qubit_is_just_h(self, isa):
        result = qft(isa, [0])
        assert len(result) == 1
        assert result[0].symbol == "h"
        assert result[0].target_qubits == [0]

    def test_single_qubit_no_swap(self, isa):
        result = qft(isa, [0])
        assert count(result, "swap") == 0


class TestQFTTwoQubits:
    def test_default_with_swaps(self, isa):
        # MSB-down: H q1; CP q0->q1; H q0; SWAP q0,q1
        result = qft(isa, [0, 1])
        assert symbols(result) == ["h", "cp", "h", "swap"]

    def test_no_swap_option(self, isa):
        result = qft(isa, [0, 1], do_swaps=False)
        assert symbols(result) == ["h", "cp", "h"]

    def test_phase_control_and_target(self, isa):
        result = qft(isa, [0, 1])
        cp = [i for i in result if i.symbol == "cp"][0]
        assert cp.control_qubits == [0]
        assert cp.target_qubits == [1]
        assert math.isclose(cp.params[0], math.pi / 2)


class TestQFTThreeQubits:
    def test_structure(self, isa):
        # MSB-down: H q2; CP q1->q2 (pi/2); CP q0->q2 (pi/4);
        # H q1; CP q0->q1 (pi/2); H q0; SWAP q0,q2
        result = qft(isa, [0, 1, 2])
        assert symbols(result) == [
            "h", "cp", "cp", "h", "cp", "h", "swap"
        ]

    def test_phase_angles_and_directions(self, isa):
        result = qft(isa, [0, 1, 2])
        cps = [i for i in result if i.symbol == "cp"]
        # cp[0]: q1 -> q2, pi/2
        assert cps[0].control_qubits == [1] and cps[0].target_qubits == [2]
        assert math.isclose(cps[0].params[0], math.pi / 2)
        # cp[1]: q0 -> q2, pi/4
        assert cps[1].control_qubits == [0] and cps[1].target_qubits == [2]
        assert math.isclose(cps[1].params[0], math.pi / 4)
        # cp[2]: q0 -> q1, pi/2
        assert cps[2].control_qubits == [0] and cps[2].target_qubits == [1]
        assert math.isclose(cps[2].params[0], math.pi / 2)

    def test_swap_targets_outermost(self, isa):
        result = qft(isa, [0, 1, 2])
        sw = [i for i in result if i.symbol == "swap"]
        assert len(sw) == 1
        assert set([sw[0].control_qubits[0], sw[0].target_qubits[0]]) == {0, 2}


class TestQFTGateCounts:
    def test_four_qubits(self, isa):
        # n H + n(n-1)/2 CP + n/2 SWAP = 4 + 6 + 2 = 12
        result = qft(isa, [0, 1, 2, 3])
        assert count(result, "h") == 4
        assert count(result, "cp") == 6
        assert count(result, "swap") == 2


class TestQFTNonContiguous:
    def test_arbitrary_qubit_indices(self, isa):
        result = qft(isa, [5, 2, 9])
        hs = [i for i in result if i.symbol == "h"]
        # MSB-down on [5, 2, 9] processes qubits[2]=9, qubits[1]=2, qubits[0]=5
        assert [h.target_qubits[0] for h in hs] == [9, 2, 5]


class TestQFTEndianness:
    def test_big_reverses_iteration(self, isa):
        # endianness="big" reverses target before iteration. For target [0,1,2],
        # reversed = [2,1,0]; MSB-down loops qubits[2]=0, qubits[1]=1, qubits[0]=2
        result = qft(isa, [0, 1, 2], endianness="big")
        hs = [i for i in result if i.symbol == "h"]
        assert [h.target_qubits[0] for h in hs] == [0, 1, 2]

    def test_big_invalid_value_raises(self, isa):
        with pytest.raises(ValueError, match="endianness"):
            qft(isa, [0, 1], endianness="middle")


class TestQFTValidation:
    def test_empty_target_raises(self, isa):
        with pytest.raises(ValueError, match="at least 1 qubit"):
            qft(isa, [])


# ===========================================================================
# qft — numerical correctness vs. textbook DFT
# ===========================================================================

class TestQFTSemantics:
    """Verify qft (with swaps) implements the standard DFT on basis states:
        QFT|b> = (1/sqrt(N)) sum_x exp(2*pi*i * x*b / N) |x>
    in little-endian convention (qubit 0 = LSB)."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_basis_states_match_dft(self, isa, n):
        N = 1 << n
        target = list(range(n))
        circuit = qft(isa, target)
        for b in range(N):
            psi = simulate(circuit, n, basis_state(b, n))
            expected = np.array(
                [np.exp(2j * np.pi * x * b / N) / np.sqrt(N) for x in range(N)]
            )
            assert np.allclose(psi, expected, atol=1e-9), (
                f"QFT|{b}> mismatch at n={n}"
            )


# ===========================================================================
# iqft — structural correctness and inverse relationship
# ===========================================================================

class TestIQFTSingleQubit:
    def test_single_qubit_is_just_h(self, isa):
        result = iqft(isa, [0])
        assert len(result) == 1
        assert result[0].symbol == "h"


class TestIQFTTwoQubits:
    def test_default_with_swaps(self, isa):
        # SWAP q0,q1; H q0; CP q0->q1 (-pi/2); H q1
        result = iqft(isa, [0, 1])
        assert symbols(result) == ["swap", "h", "cp", "h"]

    def test_no_swap_option(self, isa):
        result = iqft(isa, [0, 1], do_swaps=False)
        assert symbols(result) == ["h", "cp", "h"]

    def test_phase_angle_is_negated(self, isa):
        result = iqft(isa, [0, 1])
        cp = [i for i in result if i.symbol == "cp"][0]
        assert cp.control_qubits == [0]
        assert cp.target_qubits == [1]
        assert math.isclose(cp.params[0], -math.pi / 2)


class TestIQFTInverseOfQFT:
    def test_gate_counts_match(self, isa):
        for n in (1, 2, 3, 4):
            target = list(range(n))
            fwd = qft(isa, target)
            inv = iqft(isa, target)
            assert count(fwd, "h") == count(inv, "h")
            assert count(fwd, "cp") == count(inv, "cp")
            assert count(fwd, "swap") == count(inv, "swap")

    def test_iqft_is_reverse_of_qft_with_negated_angles(self, isa):
        for n in (1, 2, 3, 4):
            target = list(range(n))
            fwd = qft(isa, target, do_swaps=False)
            inv = iqft(isa, target, do_swaps=False)
            assert len(fwd) == len(inv)
            for f, i in zip(fwd, reversed(inv)):
                assert f.symbol == i.symbol
                assert f.target_qubits == i.target_qubits
                assert f.control_qubits == i.control_qubits
                if f.symbol == "cp":
                    assert math.isclose(f.params[0], -i.params[0])

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_qft_then_iqft_is_identity(self, isa, n):
        target = list(range(n))
        circuit = qft(isa, target) + iqft(isa, target)
        for b in range(1 << n):
            psi = simulate(circuit, n, basis_state(b, n))
            expected = basis_state(b, n)
            assert np.allclose(psi, expected, atol=1e-9)


class TestIQFTValidation:
    def test_empty_target_raises(self, isa):
        with pytest.raises(ValueError, match="at least 1 qubit"):
            iqft(isa, [])

    def test_invalid_endianness_raises(self, isa):
        with pytest.raises(ValueError, match="endianness"):
            iqft(isa, [0, 1], endianness="middle")
