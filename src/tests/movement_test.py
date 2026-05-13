"""
Filename: movement_test.py
Author: Santiago Nunez-Corrales
Date: 2026-05-12
Version: 1.0
Description:
    Tests for movement primitives: swap and entangle_step.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest
from lccfq_lang.arch.isa import ISA
from lccfq_lang.lang.movement import swap, entangle_step


@pytest.fixture
def isa():
    return ISA("test")


def ops(instrs):
    return [(i.symbol, i.control_qubits, i.target_qubits) for i in instrs]


def symbols(instrs):
    return [i.symbol for i in instrs]


# ===========================================================================
# swap
# ===========================================================================

class TestSwapPair:
    def test_basic(self, isa):
        result = swap(isa, [0, 1])
        assert len(result) == 1
        assert result[0].symbol == "swap"
        assert result[0].control_qubits == [0]
        assert result[0].target_qubits == [1]

    def test_non_adjacent(self, isa):
        result = swap(isa, [2, 5])
        assert len(result) == 1
        assert result[0].symbol == "swap"
        assert result[0].control_qubits == [2]
        assert result[0].target_qubits == [5]

    def test_rejects_single_target(self, isa):
        with pytest.raises(ValueError, match="requires len\\(target\\) == 2"):
            swap(isa, [0])

    def test_rejects_three_targets(self, isa):
        with pytest.raises(ValueError, match="requires len\\(target\\) == 2"):
            swap(isa, [0, 1, 2])


class TestSwapPairs:
    def test_two_independent_swaps(self, isa):
        result = swap(isa, [0, 1, 2, 3], pairs=[(0, 1), (2, 3)])
        assert symbols(result) == ["swap", "swap"]
        assert result[0].control_qubits == [0]
        assert result[0].target_qubits == [1]
        assert result[1].control_qubits == [2]
        assert result[1].target_qubits == [3]

    def test_empty_pairs(self, isa):
        assert swap(isa, [0, 1], pairs=[]) == []

    def test_rejects_qubit_outside_target(self, isa):
        with pytest.raises(ValueError, match="not a subset"):
            swap(isa, [0, 1], pairs=[(0, 2)])

    def test_rejects_self_swap(self, isa):
        with pytest.raises(ValueError, match="distinct"):
            swap(isa, [0, 1], pairs=[(0, 0)])


# ===========================================================================
# entangle_step
# ===========================================================================

def cx_pairs(instrs):
    return [(i.symbol, i.control_qubits[0], i.target_qubits[0]) for i in instrs]


class TestEntangleStepLinear:
    def test_default_is_linear(self, isa):
        result = entangle_step(isa, [0, 1, 2, 3])
        assert cx_pairs(result) == [
            ("cx", 0, 1),
            ("cx", 1, 2),
            ("cx", 2, 3),
        ]

    def test_two_qubits(self, isa):
        result = entangle_step(isa, [0, 1])
        assert cx_pairs(result) == [("cx", 0, 1)]

    def test_non_contiguous_indices(self, isa):
        result = entangle_step(isa, [5, 2, 7], topology="linear")
        assert cx_pairs(result) == [("cx", 5, 2), ("cx", 2, 7)]


class TestEntangleStepRing:
    def test_three_qubits(self, isa):
        result = entangle_step(isa, [0, 1, 2], topology="ring")
        assert cx_pairs(result) == [("cx", 0, 1), ("cx", 1, 2), ("cx", 2, 0)]

    def test_two_qubits_no_extra_wrap(self, isa):
        result = entangle_step(isa, [0, 1], topology="ring")
        assert cx_pairs(result) == [("cx", 0, 1)]


class TestEntangleStepPairs:
    def test_four_qubits(self, isa):
        result = entangle_step(isa, [0, 1, 2, 3], topology="pairs")
        assert cx_pairs(result) == [("cx", 0, 1), ("cx", 2, 3)]

    def test_odd_count_drops_last(self, isa):
        result = entangle_step(isa, [0, 1, 2, 3, 4], topology="pairs")
        assert cx_pairs(result) == [("cx", 0, 1), ("cx", 2, 3)]


class TestEntangleStepBrickwork:
    def test_four_qubits(self, isa):
        result = entangle_step(isa, [0, 1, 2, 3], topology="brickwork")
        assert cx_pairs(result) == [
            ("cx", 0, 1),
            ("cx", 2, 3),
            ("cx", 1, 2),
        ]


class TestEntangleStepAllToAll:
    def test_three_qubits(self, isa):
        result = entangle_step(isa, [0, 1, 2], topology="all_to_all")
        assert cx_pairs(result) == [("cx", 0, 1), ("cx", 0, 2), ("cx", 1, 2)]

    def test_four_qubits_count(self, isa):
        result = entangle_step(isa, [0, 1, 2, 3], topology="all_to_all")
        assert len(result) == 6


class TestEntangleStepGateChoice:
    def test_cz_gate(self, isa):
        result = entangle_step(isa, [0, 1, 2], gate="cz")
        assert all(i.symbol == "cz" for i in result)

    def test_invalid_gate_raises(self, isa):
        with pytest.raises(ValueError, match="gate must be"):
            entangle_step(isa, [0, 1], gate="crx")


class TestEntangleStepValidation:
    def test_single_qubit_raises(self, isa):
        with pytest.raises(ValueError, match="at least 2 qubits"):
            entangle_step(isa, [0])

    def test_empty_target_raises(self, isa):
        with pytest.raises(ValueError, match="at least 2 qubits"):
            entangle_step(isa, [])

    def test_unknown_topology_raises(self, isa):
        with pytest.raises(ValueError, match="Unknown topology"):
            entangle_step(isa, [0, 1], topology="star")
