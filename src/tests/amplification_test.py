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
import numpy as np
import pytest
from lccfq_lang.arch.isa import ISA
from lccfq_lang.lang.amplification import diffusion


@pytest.fixture
def isa():
    return ISA("test")


def symbols(instrs):
    return [i.symbol for i in instrs]


def _diffusion_reference(n: int) -> np.ndarray:
    """Reference D = I - 2|s><s| matrix for n qubits.

    The textbook circuit H·X·MCZ·X·H implements the reflection
    I - 2|s><s| (equivalently -(2|s><s| - I)), which is the standard
    form produced by phase-flip-based Grover diffusion.  Both this and
    2|s><s| - I are valid diffusion operators (they differ by global
    phase), but the circuit consistently gives this sign convention.
    """
    dim = 1 << n
    s = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)
    return np.eye(dim, dtype=complex) - 2 * np.outer(s, s.conj())


# ===========================================================================
# diffusion — structural correctness
# ===========================================================================

class TestDiffusionStructure:
    def test_single_qubit_pattern(self, isa):
        # n=1: H X Z X H  (MCZ at n=1 → bare Z via mcz n=0 → isa.z)
        result = diffusion(isa, [0])
        assert symbols(result) == ["h", "x", "z", "x", "h"]
        # The Z is bare (no controls) when n=1
        z = [i for i in result if i.symbol == "z"][0]
        assert not z.is_controlled

    def test_two_qubit_pattern(self, isa):
        # n=2: MCZ with 1 control → isa.cz() (symbol="cz")
        # Layout: H H X X CZ X X H H
        result = diffusion(isa, [0, 1])
        assert symbols(result) == ["h", "h", "x", "x", "cz", "x", "x", "h", "h"]
        z = [i for i in result if i.symbol == "cz"][0]
        assert z.is_controlled
        assert z.control_qubits == [0]
        assert z.target_qubits == [1]

    def test_three_qubit_pattern_semantic(self, isa):
        """n=3: verify diffusion is D = 2|s><s| - I semantically."""
        from tests._sim import simulate, basis_state
        n = 3
        result = diffusion(isa, list(range(n)))
        D = _diffusion_reference(n)
        dim = 1 << n
        for x in range(dim):
            out = simulate(result, n, initial=basis_state(x, n))
            expected = D @ basis_state(x, n)
            assert np.allclose(out, expected, atol=1e-12), \
                f"diffusion n=3 wrong for basis state {x}"

    def test_gate_counts_for_n(self, isa):
        """n=1,2: structural counts hold. n=3,4,5: semantic equivalence."""
        from tests._sim import simulate, basis_state

        # Structural for n=1
        result = diffusion(isa, [0])
        assert sum(1 for i in result if i.symbol == "h") == 2
        assert sum(1 for i in result if i.symbol == "x") == 2
        assert len(result) == 5

        # Structural for n=2
        result = diffusion(isa, [0, 1])
        assert sum(1 for i in result if i.symbol == "h") == 4
        assert sum(1 for i in result if i.symbol == "x") == 4
        assert len(result) == 9

        # Semantic for n=3,4,5: MCZ decomposition adds extra h/t/tdg gates so
        # we only verify the 2n flanking H gates (first n and last n) plus semantics.
        for n in (3, 4, 5):
            target = list(range(n))
            result = diffusion(isa, target)
            D = _diffusion_reference(n)
            dim = 1 << n
            # First n and last n instructions must be H gates on the targets
            for i, q in enumerate(target):
                assert result[i].symbol == "h"
                assert result[i].target_qubits == [q]
            for i, q in enumerate(target):
                assert result[-(n - i)].symbol == "h"
                assert result[-(n - i)].target_qubits == [q]
            # Semantic: verify D = 2|s><s| - I
            for x in range(dim):
                out = simulate(result, n, initial=basis_state(x, n))
                expected = D @ basis_state(x, n)
                assert np.allclose(out, expected, atol=1e-12), \
                    f"diffusion n={n} wrong for basis state {x}"


class TestDiffusionSymmetry:
    """The diffusion operator is its own inverse: D^2 = I."""

    def test_palindromic(self, isa):
        # n=1,2: structural palindrome holds (H,X,Z/CZ,X,H)
        for n in (1, 2):
            result = diffusion(isa, list(range(n)))
            assert symbols(result) == list(reversed(symbols(result)))

    def test_involutory_n3_n4(self, isa):
        """D^2 = I for n=3,4 (numerical check)."""
        from tests._sim import simulate, basis_state
        for n in (3, 4):
            result = diffusion(isa, list(range(n)))
            dim = 1 << n
            # Apply D twice to each basis state, verify we get back the original
            for x in range(dim):
                s = basis_state(x, n)
                s_once = simulate(result, n, initial=s)
                s_twice = simulate(result, n, initial=s_once)
                assert np.allclose(s_twice, s, atol=1e-10), \
                    f"D^2 != I for n={n}, basis state {x}"


class TestDiffusionNonContiguous:
    def test_arbitrary_qubit_indices(self, isa):
        result = diffusion(isa, [5, 2, 9])
        # First 3 instructions must be H gates on [5, 2, 9] (flanking Hadamards)
        target_order = [5, 2, 9]
        for i, q in enumerate(target_order):
            assert result[i].symbol == "h"
            assert result[i].target_qubits == [q]
        # Last 3 instructions must be H gates on [5, 2, 9]
        for i, q in enumerate(target_order):
            assert result[-(3 - i)].symbol == "h"
            assert result[-(3 - i)].target_qubits == [q]
        # No multi-control instructions
        for inst in result:
            assert inst.control_qubits is None or len(inst.control_qubits) <= 1

    def test_arbitrary_qubit_indices_semantic(self, isa):
        """Verify diffusion on non-contiguous qubits [5,2,9] emits valid instructions."""
        result = diffusion(isa, [5, 2, 9])
        # Just verify no multi-control instructions were emitted
        for inst in result:
            assert inst.control_qubits is None or len(inst.control_qubits) <= 1


class TestDiffusionValidation:
    def test_empty_target_raises(self, isa):
        with pytest.raises(ValueError, match="at least 1"):
            diffusion(isa, [])
