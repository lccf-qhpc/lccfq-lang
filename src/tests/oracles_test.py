"""
Filename: oracles_test.py
Author: Santiago Nunez-Corrales
Date: 2026-05-12
Version: 1.0
Description:
    Tests for oracle primitives: oracle (bit-flip) and phase_oracle.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import numpy as np
import pytest
from lccfq_lang.arch.isa import ISA
from lccfq_lang.lang.oracles import oracle, phase_oracle


@pytest.fixture
def isa():
    return ISA("test")


def symbols(instrs):
    return [i.symbol for i in instrs]


def count_mcx_like(instrs, ancilla):
    """Count controlled-X style gates targeting ancilla (cx or old-style x controlled)."""
    return sum(
        1 for i in instrs
        if i.symbol in ("x", "cx")
        and i.is_controlled
        and i.target_qubits == [ancilla]
    )


# ===========================================================================
# oracle — basic marking patterns
# ===========================================================================

class TestOracleEmptyMarking:
    def test_no_marked_inputs(self, isa):
        result = oracle(isa, [0, 1], ancilla=2, predicate=[])
        assert result == []

    def test_callable_always_false(self, isa):
        result = oracle(isa, [0, 1], ancilla=2, predicate=lambda x: False)
        assert result == []


class TestOracleSingleInput:
    """n=1: target has one qubit. After refactor, the MCX at n=1 is a CX gate."""

    def test_mark_one(self, isa):
        result = oracle(isa, [0], ancilla=1, predicate=[1])
        # Single controlled-X: isa.cx returns symbol="cx"
        assert len(result) == 1
        i = result[0]
        assert i.symbol == "cx"
        assert i.is_controlled
        assert i.control_qubits == [0]
        assert i.target_qubits == [1]

    def test_mark_zero(self, isa):
        result = oracle(isa, [0], ancilla=1, predicate=[0])
        # X q0; CX q0->q1; X q0
        assert len(result) == 3
        assert result[0].symbol == "x" and not result[0].is_controlled
        assert result[1].is_controlled and result[1].control_qubits == [0]
        assert result[2].symbol == "x" and not result[2].is_controlled

    def test_mark_both(self, isa):
        result = oracle(isa, [0], ancilla=1, predicate=[0, 1])
        # Two MCX gates targeting ancilla=1
        assert count_mcx_like(result, ancilla=1) == 2


class TestOracleTwoInputs:
    """n=2: MCX with 2 controls now decomposes to Toffoli (15 instructions).
    We keep semantic checks (is_controlled on the target) and verify the
    decomposition is correct via simulator."""

    def test_mark_value_3_emits_controlled_block(self, isa):
        # 3 = '11' → no surrounding X needed; full Toffoli expansion to ancilla=2
        result = oracle(isa, [0, 1], ancilla=2, predicate=[3])
        # Toffoli is 15 gates. Must be non-empty.
        assert len(result) > 1
        # All instructions must be 1q or 2q (no multi-control)
        for inst in result:
            assert inst.control_qubits is None or len(inst.control_qubits) <= 1

    def test_mark_value_3_semantic(self, isa):
        from tests._sim import simulate, basis_state
        # |11>|0> -> |11>|1>
        n = 3  # 2 inputs + 1 ancilla
        result = oracle(isa, [0, 1], ancilla=2, predicate=[3])
        # Apply to |110> (qubits: ancilla=q2=0, target[1]=q1=1, target[0]=q0=1)
        # Little-endian: index = q0 + 2*q1 + 4*q2 = 1 + 2 + 0 = 3 = |011> in index
        s_in = basis_state(3, n)   # q0=1, q1=1, q2=0 → index 3
        s_out = simulate(result, n, initial=s_in)
        # ancilla should flip: index = 1+2+4 = 7
        assert np.allclose(s_out, basis_state(7, n), atol=1e-12)

    def test_mark_value_0_wraps_both_with_x(self, isa):
        # 0 = '00' → flip both input qubits before and after the MCX
        result = oracle(isa, [0, 1], ancilla=2, predicate=[0])
        # Starts with 2 X gates, ends with 2 X gates, MCX in between (expanded)
        assert result[0].symbol == "x" and not result[0].is_controlled
        assert result[1].symbol == "x" and not result[1].is_controlled
        assert result[-1].symbol == "x" and not result[-1].is_controlled
        assert result[-2].symbol == "x" and not result[-2].is_controlled
        # No multi-control instructions in the list
        for inst in result:
            assert inst.control_qubits is None or len(inst.control_qubits) <= 1

    def test_mark_value_1_wraps_only_high_bit(self, isa):
        # 1 = bit 0 set, bit 1 clear (little-endian default) → flip q1 only
        result = oracle(isa, [0, 1], ancilla=2, predicate=[1])
        assert result[0].symbol == "x" and result[0].target_qubits == [1]
        assert result[-1].symbol == "x" and result[-1].target_qubits == [1]
        # No multi-control instructions
        for inst in result:
            assert inst.control_qubits is None or len(inst.control_qubits) <= 1

    def test_bitstring_and_int_agree(self, isa):
        # "01" reads MSB-first, so int = 0b01 = 1
        from tests._sim import simulate, basis_state
        n = 3
        result_str = oracle(isa, [0, 1], ancilla=2, predicate=["01"])
        result_int = oracle(isa, [0, 1], ancilla=2, predicate=[1])
        # Both should produce identical semantic results
        for x in range(4):
            s = basis_state(x, n)
            out_str = simulate(result_str, n, initial=s)
            out_int = simulate(result_int, n, initial=s)
            assert np.allclose(out_str, out_int, atol=1e-12)

    def test_endianness_big_swaps_bit_meaning(self, isa):
        # "01" with big endianness: target[0] is MSB, so bit-0=0, bit-1=1
        result = oracle(isa, [0, 1], ancilla=2, predicate=["01"],
                        endianness="big")
        # The X-wrap should be on q0 (the bit that is 0)
        assert result[0].target_qubits == [0]
        assert result[-1].target_qubits == [0]


class TestOracleThreeInputs:
    def test_three_inputs_semantic(self, isa):
        """n=3: verify oracle semantics via simulation rather than structure."""
        from tests._sim import simulate, basis_state
        n_in, n_anc = 3, 1
        n = n_in + n_anc
        marked = (0, 3, 5, 7)
        result = oracle(isa, [0, 1, 2], ancilla=3,
                        predicate=lambda x: x in marked)
        # For each input |x>|0>, oracle should give |x>|f(x)>
        for x in range(8):
            s_init = basis_state(x, n)   # |x>|0>, ancilla at qubit 3 = 0
            s_out = simulate(result, n, initial=s_init)
            expected_idx = x | ((1 if x in marked else 0) << n_in)
            expected = basis_state(expected_idx, n)
            assert np.allclose(s_out, expected, atol=1e-12), \
                f"Failed for x={x}: expected index {expected_idx}"

    def test_no_multi_control_instructions(self, isa):
        """All emitted instructions must have at most 1 control qubit."""
        result = oracle(isa, [0, 1, 2], ancilla=3,
                        predicate=lambda x: x in (0, 3, 5, 7))
        for inst in result:
            assert inst.control_qubits is None or len(inst.control_qubits) <= 1


class TestOracleValidation:
    def test_empty_target_raises(self, isa):
        with pytest.raises(ValueError, match="at least 1"):
            oracle(isa, [], ancilla=0, predicate=[])

    def test_ancilla_overlaps_target(self, isa):
        with pytest.raises(ValueError, match="ancilla.*must not appear"):
            oracle(isa, [0, 1], ancilla=1, predicate=[3])

    def test_int_out_of_range(self, isa):
        with pytest.raises(ValueError, match="out of range"):
            oracle(isa, [0, 1], ancilla=2, predicate=[5])

    def test_bitstring_wrong_length(self, isa):
        with pytest.raises(ValueError, match="length 2"):
            oracle(isa, [0, 1], ancilla=2, predicate=["101"])

    def test_bitstring_bad_char(self, isa):
        with pytest.raises(ValueError, match="length 2"):
            oracle(isa, [0, 1], ancilla=2, predicate=["1x"])

    def test_invalid_endianness(self, isa):
        with pytest.raises(ValueError, match="endianness"):
            oracle(isa, [0, 1], ancilla=2, predicate=[],
                   endianness="middle")

    def test_predicate_item_wrong_type(self, isa):
        with pytest.raises(TypeError):
            oracle(isa, [0, 1], ancilla=2, predicate=[1.5])


# ===========================================================================
# phase_oracle — ancilla-free phase-kickback
# ===========================================================================

class TestPhaseOracleEmpty:
    def test_no_marked_inputs(self, isa):
        result = phase_oracle(isa, [0, 1], predicate=[])
        assert result == []


class TestPhaseOracleSingleQubit:
    def test_mark_one_is_just_z(self, isa):
        result = phase_oracle(isa, [0], predicate=[1])
        assert len(result) == 1
        assert result[0].symbol == "z"
        assert not result[0].is_controlled
        assert result[0].target_qubits == [0]

    def test_mark_zero_wraps_z_in_x(self, isa):
        result = phase_oracle(isa, [0], predicate=[0])
        # X q0; Z q0; X q0
        assert symbols(result) == ["x", "z", "x"]
        assert result[1].target_qubits == [0]


class TestPhaseOracleTwoQubits:
    """n=2: MCZ at n=1 control now returns isa.cz() with symbol='cz'."""

    def test_mark_three_is_cz(self, isa):
        result = phase_oracle(isa, [0, 1], predicate=[3])
        assert len(result) == 1
        mcz = result[0]
        assert mcz.symbol == "cz"
        assert mcz.is_controlled
        assert mcz.control_qubits == [0]
        assert mcz.target_qubits == [1]

    def test_mark_zero_wraps_both(self, isa):
        result = phase_oracle(isa, [0, 1], predicate=[0])
        # X q0; X q1; CZ q0->q1; X q0; X q1
        assert len(result) == 5
        assert symbols(result) == ["x", "x", "cz", "x", "x"]
        assert result[2].is_controlled
        assert result[2].control_qubits == [0]
        assert result[2].target_qubits == [1]

    def test_endianness_big(self, isa):
        # "01" big-endian: target[0] is MSB so x = 01 means bit-0=0, bit-1=1
        # → X wrap on q0 only
        result = phase_oracle(isa, [0, 1], predicate=["01"],
                              endianness="big")
        assert len(result) == 3
        assert result[0].target_qubits == [0]
        assert result[2].target_qubits == [0]


class TestPhaseOracleThreeQubits:
    def test_three_qubits_semantic(self, isa):
        """n=3: verify phase oracle semantics via simulation."""
        from tests._sim import simulate, basis_state
        n = 3
        marked = (7,)
        result = phase_oracle(isa, [0, 1, 2], predicate=lambda x: x in marked)
        for x in range(1 << n):
            s = simulate(result, n, initial=basis_state(x, n))
            sign = -1.0 if x in marked else 1.0
            assert np.allclose(s, sign * basis_state(x, n), atol=1e-12), \
                f"Failed for x={x}"

    def test_count_only_mcz_semantic(self, isa):
        """n=3 with multiple marked inputs: verify via simulation."""
        from tests._sim import simulate, basis_state
        n = 3
        marked = (0, 3, 5, 7)
        result = phase_oracle(
            isa, [0, 1, 2], predicate=lambda x: x in marked
        )
        for x in range(1 << n):
            s = simulate(result, n, initial=basis_state(x, n))
            sign = -1.0 if x in marked else 1.0
            assert np.allclose(s, sign * basis_state(x, n), atol=1e-12), \
                f"Failed for x={x}"

    def test_no_multi_control_instructions(self, isa):
        """All emitted instructions must have at most 1 control qubit."""
        result = phase_oracle(isa, [0, 1, 2], predicate=[7])
        for inst in result:
            assert inst.control_qubits is None or len(inst.control_qubits) <= 1


class TestPhaseOracleValidation:
    def test_empty_target_raises(self, isa):
        with pytest.raises(ValueError, match="at least 1"):
            phase_oracle(isa, [], predicate=[])

    def test_invalid_endianness(self, isa):
        with pytest.raises(ValueError, match="endianness"):
            phase_oracle(isa, [0, 1], predicate=[], endianness="middle")

    def test_int_out_of_range(self, isa):
        with pytest.raises(ValueError, match="out of range"):
            phase_oracle(isa, [0, 1], predicate=[5])

    def test_bitstring_wrong_length(self, isa):
        with pytest.raises(ValueError, match="length 2"):
            phase_oracle(isa, [0, 1], predicate=["101"])
