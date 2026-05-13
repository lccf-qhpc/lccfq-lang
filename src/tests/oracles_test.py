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
import pytest
from lccfq_lang.arch.isa import ISA
from lccfq_lang.lang.oracles import oracle, phase_oracle


@pytest.fixture
def isa():
    return ISA("test")


def symbols(instrs):
    return [i.symbol for i in instrs]


def count_mcx(instrs, ancilla):
    return sum(
        1 for i in instrs
        if i.symbol == "x"
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
    """n=1: target has one qubit. Marking 0 surrounds the CX with X gates;
    marking 1 leaves the CX bare."""

    def test_mark_one(self, isa):
        result = oracle(isa, [0], ancilla=1, predicate=[1])
        # Only the multi-controlled-X (single control here)
        assert len(result) == 1
        i = result[0]
        assert i.symbol == "x"
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
        # Two patterns, total = 1 (mark 0: X-CX-X) + 1 (mark 1: CX) = 4 instructions
        assert count_mcx(result, ancilla=1) == 2


class TestOracleTwoInputs:
    def test_mark_value_3_emits_bare_mcx(self, isa):
        # 3 = '11' → no surrounding X needed
        result = oracle(isa, [0, 1], ancilla=2, predicate=[3])
        assert len(result) == 1
        mcx = result[0]
        assert mcx.is_controlled
        assert mcx.control_qubits == [0, 1]
        assert mcx.target_qubits == [2]

    def test_mark_value_0_wraps_both_with_x(self, isa):
        # 0 = '00' → flip both input qubits before and after the MCX
        result = oracle(isa, [0, 1], ancilla=2, predicate=[0])
        # Expected order: X q0, X q1, MCX q0,q1->q2, X q0, X q1
        assert len(result) == 5
        assert symbols(result) == ["x", "x", "x", "x", "x"]
        # The middle one is the controlled one
        assert result[2].is_controlled
        assert result[2].control_qubits == [0, 1]

    def test_mark_value_1_wraps_only_high_bit(self, isa):
        # 1 = bit 0 set, bit 1 clear (little-endian default) → flip q1 only
        result = oracle(isa, [0, 1], ancilla=2, predicate=[1])
        assert len(result) == 3
        assert result[0].symbol == "x" and result[0].target_qubits == [1]
        assert result[1].is_controlled
        assert result[2].symbol == "x" and result[2].target_qubits == [1]

    def test_bitstring_and_int_agree(self, isa):
        # "01" reads MSB-first, so int = 0b01 = 1 → little-endian: bit-0=1, bit-1=0
        # Same as predicate=[1]: should wrap q1 only.
        result = oracle(isa, [0, 1], ancilla=2, predicate=["01"])
        assert len(result) == 3
        assert result[0].target_qubits == [1]
        assert result[2].target_qubits == [1]

    def test_endianness_big_swaps_bit_meaning(self, isa):
        # "01" with big endianness: target[0] is MSB, so bit-0=0, bit-1=1
        # The X-wrap should be on q0 (the bit that is 0).
        result = oracle(isa, [0, 1], ancilla=2,
                        predicate=["01"], endianness="big")
        assert len(result) == 3
        assert result[0].target_qubits == [0]
        assert result[2].target_qubits == [0]


class TestOracleThreeInputs:
    def test_count_only_mcx(self, isa):
        # 4 marked inputs → 4 MCX gates
        result = oracle(isa, [0, 1, 2], ancilla=3,
                        predicate=lambda x: x in (0, 3, 5, 7))
        assert count_mcx(result, ancilla=3) == 4


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

def find_mcz(instrs, n_targets):
    """Return the MCZ-like instruction in instrs (n=1: Z; n=2: CZ;
    n>=3: multi-controlled Z)."""
    for i in instrs:
        if i.symbol == "z" and (
            (n_targets == 1 and not i.is_controlled)
            or (n_targets >= 2 and i.is_controlled
                and len(i.control_qubits) == n_targets - 1)
        ):
            return i
    return None


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
    def test_mark_three_is_cz(self, isa):
        result = phase_oracle(isa, [0, 1], predicate=[3])
        assert len(result) == 1
        mcz = result[0]
        assert mcz.symbol == "z"
        assert mcz.is_controlled
        assert mcz.control_qubits == [0]
        assert mcz.target_qubits == [1]

    def test_mark_zero_wraps_both(self, isa):
        result = phase_oracle(isa, [0, 1], predicate=[0])
        # X q0; X q1; CZ q0->q1; X q0; X q1
        assert len(result) == 5
        assert symbols(result) == ["x", "x", "z", "x", "x"]
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
    def test_mcz_has_two_controls(self, isa):
        result = phase_oracle(isa, [0, 1, 2], predicate=[7])
        assert len(result) == 1
        mcz = result[0]
        assert mcz.symbol == "z"
        assert mcz.is_controlled
        assert mcz.control_qubits == [0, 1]
        assert mcz.target_qubits == [2]

    def test_count_only_mcz(self, isa):
        result = phase_oracle(
            isa, [0, 1, 2], predicate=lambda x: x in (0, 3, 5, 7)
        )
        mczs = [i for i in result if i.symbol == "z"]
        assert len(mczs) == 4
        assert all(
            i.is_controlled and i.control_qubits == [0, 1]
            for i in mczs
        )


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
