"""
Filename: arithmetic_test.py
Author: Santiago Nunez-Corrales
Date: 2026-05-12
Version: 1.0
Description:
    Tests for quantum arithmetic primitives: add, compare, mult_mod.
    Uses a small dense-vector simulator to verify end-to-end correctness on
    n-bit registers for n up to ~4. Larger cases are out of scope.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import numpy as np
import pytest
from lccfq_lang.arch.isa import ISA
from lccfq_lang.lang.arithmetic import add, compare, mult_mod

from ._sim import simulate, basis_state as _basis_state


@pytest.fixture
def isa():
    return ISA("test")


# ===========================================================================
# add — classical + quantum
# ===========================================================================

class TestAddClassical:
    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_all_pairs_modular(self, isa, n):
        dim = 1 << n
        for b in range(dim):
            for a in range(dim):
                circuit = add(isa, list(range(n)), value=a)
                psi0 = _basis_state(b, n)
                psi1 = simulate(circuit, n, psi0)
                expected = (b + a) % dim
                got = int(np.argmax(np.abs(psi1) ** 2))
                assert got == expected, (
                    f"n={n} b={b} a={a} expected {expected} got {got}"
                )
                # Peak amplitude near 1 (allow phase + numerical noise)
                assert np.abs(psi1[expected]) > 1 - 1e-9

    def test_zero_addend_is_identity(self, isa):
        n = 3
        for b in range(1 << n):
            circuit = add(isa, list(range(n)), value=0)
            psi0 = _basis_state(b, n)
            psi1 = simulate(circuit, n, psi0)
            assert np.argmax(np.abs(psi1) ** 2) == b

    def test_wraparound_negative_value(self, isa):
        # value=-1 mod 4 should add 3
        n = 2
        circuit = add(isa, list(range(n)), value=-1)
        psi0 = _basis_state(1, n)
        psi1 = simulate(circuit, n, psi0)
        # 1 + (-1) mod 4 = 0
        assert np.argmax(np.abs(psi1) ** 2) == 0


# ===========================================================================
# add — quantum + quantum
# ===========================================================================

class TestAddQuantum:
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_all_pairs_modular(self, isa, n):
        dim = 1 << n
        # target qubits 0..n-1, register qubits n..2n-1
        target = list(range(n))
        register = list(range(n, 2 * n))
        for a in range(dim):
            for b in range(dim):
                # Compose |a>|b> with a on register, b on target (little-endian)
                combined = (a << n) | b
                psi0 = _basis_state(combined, 2 * n)
                circuit = add(isa, target, register=register)
                psi1 = simulate(circuit, 2 * n, psi0)
                expected = (a << n) | ((a + b) % dim)
                got = int(np.argmax(np.abs(psi1) ** 2))
                assert got == expected, (
                    f"n={n} a={a} b={b} expected {expected:0{2*n}b} "
                    f"got {got:0{2*n}b}"
                )

    def test_register_unchanged(self, isa):
        # After |a>|b> -> |a>|a+b>, marginal on register must still be |a>
        n = 2
        target = [0, 1]
        register = [2, 3]
        circuit = add(isa, target, register=register)
        for a in range(1 << n):
            for b in range(1 << n):
                psi0 = _basis_state((a << n) | b, 2 * n)
                psi1 = simulate(circuit, 2 * n, psi0)
                # Marginalize over target bits to recover register value
                probs = np.abs(psi1) ** 2
                reg_marg = np.zeros(1 << n)
                for idx in range(1 << (2 * n)):
                    reg_marg[idx >> n] += probs[idx]
                assert np.argmax(reg_marg) == a
                assert reg_marg[a] > 1 - 1e-9


# ===========================================================================
# add — validation
# ===========================================================================

class TestAddValidation:
    def test_no_value_no_register(self, isa):
        with pytest.raises(ValueError, match="exactly one of"):
            add(isa, [0, 1])

    def test_both_value_and_register(self, isa):
        with pytest.raises(ValueError, match="exactly one of"):
            add(isa, [0, 1], value=1, register=[2, 3])

    def test_empty_target(self, isa):
        with pytest.raises(ValueError, match="at least 1"):
            add(isa, [], value=0)

    def test_register_wrong_length(self, isa):
        with pytest.raises(ValueError, match="register length"):
            add(isa, [0, 1], register=[2])

    def test_register_overlaps_target(self, isa):
        with pytest.raises(ValueError, match="disjoint"):
            add(isa, [0, 1], register=[1, 2])


# ===========================================================================
# compare — exhaustive truth-table verification
# ===========================================================================

def _measure_result_and_target(psi, n: int, target_qubits, result_q, n_total):
    """For a state ~near a basis state, recover (target_value, result_bit).

    Asserts the state is concentrated on a single computational basis index
    so the comparison's effect can be read off deterministically."""
    probs = np.abs(psi) ** 2
    idx = int(np.argmax(probs))
    assert probs[idx] > 1 - 1e-9, (
        f"state not concentrated: max prob = {probs[idx]:.6f}"
    )
    tval = 0
    for k, q in enumerate(target_qubits):
        tval |= ((idx >> q) & 1) << k
    rbit = (idx >> result_q) & 1
    return tval, rbit, idx


class TestCompareEq:
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_truth_table(self, isa, n):
        # qubits: target = 0..n-1, result = n
        target = list(range(n))
        result_q = n
        total = n + 1
        for k in range(1 << n):
            for a in range(1 << n):
                circuit = compare(isa, target, op="eq", value=k, result=result_q)
                psi0 = _basis_state(a, total)
                psi1 = simulate(circuit, total, psi0)
                tval, rbit, _ = _measure_result_and_target(
                    psi1, n, target, result_q, total
                )
                assert tval == a  # target unchanged
                assert rbit == int(a == k), (
                    f"n={n} a={a} k={k}: expected {int(a == k)}, got {rbit}"
                )

    def test_result_xor_accumulates(self, isa):
        # Start with result=|1>: equal case should clear it; unequal preserves it.
        n = 2
        target = [0, 1]
        result_q = 2
        # Pre-flip result to 1.
        for a in range(4):
            for k in range(4):
                psi0 = _basis_state(a | (1 << result_q), 3)
                circuit = compare(isa, target, op="eq", value=k, result=result_q)
                psi1 = simulate(circuit, 3, psi0)
                _, rbit, _ = _measure_result_and_target(psi1, n, target, result_q, 3)
                assert rbit == (1 ^ int(a == k))


class TestCompareNe:
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_truth_table(self, isa, n):
        target = list(range(n))
        result_q = n
        total = n + 1
        for k in range(1 << n):
            for a in range(1 << n):
                circuit = compare(isa, target, op="ne", value=k, result=result_q)
                psi1 = simulate(
                    circuit, total, _basis_state(a, total)
                )
                tval, rbit, _ = _measure_result_and_target(
                    psi1, n, target, result_q, total
                )
                assert tval == a
                assert rbit == int(a != k)


class TestCompareLt:
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_truth_table(self, isa, n):
        # qubits: target = 0..n-1, result = n, sign_ancilla = n+1
        target = list(range(n))
        result_q = n
        sign_q = n + 1
        total = n + 2
        for k in range(1 << n):
            for a in range(1 << n):
                circuit = compare(
                    isa, target, op="lt", value=k, result=result_q,
                    sign_ancilla=sign_q,
                )
                psi1 = simulate(circuit, total, _basis_state(a, total))
                tval, rbit, idx = _measure_result_and_target(
                    psi1, n, target, result_q, total
                )
                assert tval == a, "target should be restored to a"
                assert (idx >> sign_q) & 1 == 0, "sign_ancilla should be |0>"
                assert rbit == int(a < k), (
                    f"n={n} a={a} k={k}: expected {int(a < k)}, got {rbit}"
                )

    def test_k_zero_is_always_false(self, isa):
        n = 3
        target = list(range(n))
        circuit = compare(
            isa, target, op="lt", value=0, result=n, sign_ancilla=n + 1
        )
        # No-op short-circuit
        assert circuit == []

    def test_k_at_register_max_is_always_true(self, isa):
        # k >= 2**n: a < k for any a in [0, 2**n) → flip result unconditionally
        n = 2
        target = list(range(n))
        circuit = compare(
            isa, target, op="lt", value=1 << n, result=n,
            sign_ancilla=n + 1,
        )
        assert len(circuit) == 1
        assert circuit[0].symbol == "x"
        assert circuit[0].target_qubits == [n]


class TestCompareGe:
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_truth_table(self, isa, n):
        target = list(range(n))
        result_q = n
        sign_q = n + 1
        total = n + 2
        for k in range(1 << n):
            for a in range(1 << n):
                circuit = compare(
                    isa, target, op="ge", value=k, result=result_q,
                    sign_ancilla=sign_q,
                )
                psi1 = simulate(circuit, total, _basis_state(a, total))
                tval, rbit, idx = _measure_result_and_target(
                    psi1, n, target, result_q, total
                )
                assert tval == a
                assert (idx >> sign_q) & 1 == 0
                assert rbit == int(a >= k)


class TestCompareValidation:
    def test_unknown_op(self, isa):
        with pytest.raises(ValueError, match="op must be"):
            compare(isa, [0, 1], op="abc", value=0, result=2)

    def test_empty_target(self, isa):
        with pytest.raises(ValueError, match="at least 1"):
            compare(isa, [], op="eq", value=0, result=0)

    def test_result_in_target(self, isa):
        with pytest.raises(ValueError, match="result.*must not appear"):
            compare(isa, [0, 1], op="eq", value=0, result=1)

    def test_lt_missing_ancilla(self, isa):
        with pytest.raises(ValueError, match="sign_ancilla"):
            compare(isa, [0, 1], op="lt", value=1, result=2)

    def test_lt_ancilla_overlap(self, isa):
        with pytest.raises(ValueError, match="disjoint"):
            compare(
                isa, [0, 1], op="lt", value=1, result=2, sign_ancilla=0
            )


# ===========================================================================
# mult_mod — exhaustive truth-table verification on modulus 2**n
# ===========================================================================

class TestMultModTruthTable:
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_all_pairs(self, isa, n):
        # target = 0..n-1 (output), register = n..2n-1 (input |x>)
        target = list(range(n))
        register = list(range(n, 2 * n))
        total = 2 * n
        dim = 1 << n
        for a in range(dim):
            circuit = mult_mod(isa, target, value=a, register=register)
            for x in range(dim):
                # input |x>_register |0>_target  →  index = x << n
                psi0 = _basis_state(x << n, total)
                psi1 = simulate(circuit, total, psi0)
                probs = np.abs(psi1) ** 2
                got_idx = int(np.argmax(probs))
                assert probs[got_idx] > 1 - 1e-9
                got_target = got_idx & (dim - 1)
                got_reg = got_idx >> n
                assert got_reg == x, (
                    f"n={n} a={a} x={x}: register changed from {x} to {got_reg}"
                )
                expected = (a * x) % dim
                assert got_target == expected, (
                    f"n={n} a={a} x={x}: expected target {expected}, "
                    f"got {got_target}"
                )

    def test_value_zero_is_zero(self, isa):
        # a=0 maps any x to 0 in target
        n = 3
        target = list(range(n))
        register = list(range(n, 2 * n))
        circuit = mult_mod(isa, target, value=0, register=register)
        for x in range(1 << n):
            psi0 = _basis_state(x << n, 2 * n)
            psi1 = simulate(circuit, 2 * n, psi0)
            got_idx = int(np.argmax(np.abs(psi1) ** 2))
            got_target = got_idx & ((1 << n) - 1)
            assert got_target == 0

    def test_value_one_is_copy(self, isa):
        # a=1 copies x into target
        n = 3
        target = list(range(n))
        register = list(range(n, 2 * n))
        circuit = mult_mod(isa, target, value=1, register=register)
        for x in range(1 << n):
            psi0 = _basis_state(x << n, 2 * n)
            psi1 = simulate(circuit, 2 * n, psi0)
            got_idx = int(np.argmax(np.abs(psi1) ** 2))
            got_target = got_idx & ((1 << n) - 1)
            assert got_target == x

    def test_value_reduced_modulo(self, isa):
        # value=a and value=a+2^n must produce identical circuits' effects
        n = 2
        target = [0, 1]
        register = [2, 3]
        for a in range(1 << n):
            c1 = mult_mod(isa, target, value=a, register=register)
            c2 = mult_mod(isa, target, value=a + (1 << n), register=register)
            for x in range(1 << n):
                p1 = simulate(c1, 4, _basis_state(x << n, 4))
                p2 = simulate(c2, 4, _basis_state(x << n, 4))
                assert np.allclose(p1, p2, atol=1e-9)


class TestMultModValidation:
    def test_empty_target(self, isa):
        with pytest.raises(ValueError, match="at least 1"):
            mult_mod(isa, [], value=1, register=[])

    def test_register_wrong_length(self, isa):
        with pytest.raises(ValueError, match="register length"):
            mult_mod(isa, [0, 1], value=1, register=[2])

    def test_register_overlap_target(self, isa):
        with pytest.raises(ValueError, match="disjoint"):
            mult_mod(isa, [0, 1], value=1, register=[1, 2])

    def test_non_power_of_two_modulus(self, isa):
        with pytest.raises(NotImplementedError, match="modulus"):
            mult_mod(isa, [0, 1], value=1, register=[2, 3], modulus=3)
