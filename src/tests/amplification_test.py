"""
Filename: amplification_test.py
Author: Santiago Nunez-Corrales
Date: 2026-05-12
Version: 1.0
Description:
    Tests for amplitude amplification primitives: diffusion.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest
from lccfq_lang.arch.isa import ISA
from lccfq_lang.lang.amplification import diffusion


@pytest.fixture
def isa():
    return ISA("test")


def symbols(instrs):
    return [i.symbol for i in instrs]


# ===========================================================================
# diffusion — structural correctness
# ===========================================================================

class TestDiffusionStructure:
    def test_single_qubit_pattern(self, isa):
        # n=1: H X Z X H
        result = diffusion(isa, [0])
        assert symbols(result) == ["h", "x", "z", "x", "h"]
        # The Z is bare (no controls) when n=1
        z = [i for i in result if i.symbol == "z"][0]
        assert not z.is_controlled

    def test_two_qubit_pattern(self, isa):
        # n=2: H H X X CZ X X H H
        result = diffusion(isa, [0, 1])
        assert symbols(result) == ["h", "h", "x", "x", "z", "x", "x", "h", "h"]
        z = [i for i in result if i.symbol == "z"][0]
        assert z.is_controlled
        assert z.control_qubits == [0]
        assert z.target_qubits == [1]

    def test_three_qubit_pattern(self, isa):
        # n=3: H×3, X×3, MCZ(2 controls), X×3, H×3
        result = diffusion(isa, [0, 1, 2])
        assert symbols(result) == (
            ["h"] * 3 + ["x"] * 3 + ["z"] + ["x"] * 3 + ["h"] * 3
        )
        z = [i for i in result if i.symbol == "z"][0]
        assert z.is_controlled
        assert z.control_qubits == [0, 1]
        assert z.target_qubits == [2]

    def test_gate_counts_for_n(self, isa):
        # General count: 2n H + 2n X + 1 MCZ
        for n in (1, 2, 3, 4, 5):
            target = list(range(n))
            result = diffusion(isa, target)
            assert sum(1 for i in result if i.symbol == "h") == 2 * n
            assert sum(1 for i in result if i.symbol == "x") == 2 * n
            assert sum(1 for i in result if i.symbol == "z") == 1
            assert len(result) == 4 * n + 1


class TestDiffusionSymmetry:
    """The diffusion operator is its own inverse: D^2 = I. The instruction
    list it produces must therefore be a palindrome under the gate-by-gate
    inversion (H and X are self-inverse; MCZ is self-inverse)."""

    def test_palindromic(self, isa):
        for n in (1, 2, 3, 4):
            result = diffusion(isa, list(range(n)))
            # symbols read forwards == symbols read backwards
            assert symbols(result) == list(reversed(symbols(result)))


class TestDiffusionNonContiguous:
    def test_arbitrary_qubit_indices(self, isa):
        result = diffusion(isa, [5, 2, 9])
        # Hadamards land on the same qubits as targets, in order
        hs = [i for i in result if i.symbol == "h"]
        assert [h.target_qubits[0] for h in hs[:3]] == [5, 2, 9]
        assert [h.target_qubits[0] for h in hs[3:]] == [5, 2, 9]
        z = [i for i in result if i.symbol == "z"][0]
        assert z.control_qubits == [5, 2]
        assert z.target_qubits == [9]


class TestDiffusionValidation:
    def test_empty_target_raises(self, isa):
        with pytest.raises(ValueError, match="at least 1"):
            diffusion(isa, [])
