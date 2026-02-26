"""
Filename: preparation_test.py
Author: Santiago Nunez-Corrales
Date: 2026-02-26
Version: 1.0
Description:
    Tests for state preparation blocks: prepare_basis, prepare_uniform,
    prepare_state, and the _ucr helper.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import numpy as np
import pytest
from lccfq_lang.arch.isa import ISA
from lccfq_lang.lang.preparation import (
    prepare_basis,
    prepare_uniform,
    prepare_state,
    _ucr,
)


@pytest.fixture
def isa():
    return ISA("test")


# ---------------------------------------------------------------------------
# Helper to extract (symbol, target_qubits) pairs for concise assertions
# ---------------------------------------------------------------------------

def ops(instrs):
    return [(i.symbol, i.target_qubits) for i in instrs]


def symbols(instrs):
    return [i.symbol for i in instrs]


# ===========================================================================
# prepare_basis — Z basis
# ===========================================================================

class TestPrepareBasisZ:
    def test_all_zeros(self, isa):
        result = prepare_basis(isa, [0, 1, 2], bitstring="000")
        assert result == []

    def test_all_ones(self, isa):
        result = prepare_basis(isa, [0, 1, 2], bitstring="111")
        assert ops(result) == [("x", [0]), ("x", [1]), ("x", [2])]

    def test_mixed(self, isa):
        result = prepare_basis(isa, [0, 1, 2, 3], bitstring="1010")
        assert ops(result) == [("x", [0]), ("x", [2])]

    def test_single_qubit_zero(self, isa):
        assert prepare_basis(isa, [0], bitstring="0") == []

    def test_single_qubit_one(self, isa):
        result = prepare_basis(isa, [0], bitstring="1")
        assert ops(result) == [("x", [0])]


# ===========================================================================
# prepare_basis — X basis
# ===========================================================================

class TestPrepareBasisX:
    def test_all_plus(self, isa):
        result = prepare_basis(isa, [0, 1], bitstring="00", basis="X")
        assert ops(result) == [("h", [0]), ("h", [1])]

    def test_minus_plus(self, isa):
        result = prepare_basis(isa, [0, 1], bitstring="10", basis="X")
        assert ops(result) == [("x", [0]), ("h", [0]), ("h", [1])]

    def test_all_minus(self, isa):
        result = prepare_basis(isa, [0, 1], bitstring="11", basis="X")
        assert ops(result) == [("x", [0]), ("x", [1]), ("h", [0]), ("h", [1])]


# ===========================================================================
# prepare_basis — Y basis
# ===========================================================================

class TestPrepareBasisY:
    def test_all_plus_i(self, isa):
        result = prepare_basis(isa, [0, 1], bitstring="00", basis="Y")
        assert ops(result) == [
            ("h", [0]), ("s", [0]),
            ("h", [1]), ("s", [1]),
        ]

    def test_mixed_y(self, isa):
        result = prepare_basis(isa, [0, 1], bitstring="01", basis="Y")
        assert ops(result) == [
            ("x", [1]),
            ("h", [0]), ("s", [0]),
            ("h", [1]), ("s", [1]),
        ]


# ===========================================================================
# prepare_basis — endianness
# ===========================================================================

class TestPrepareBasisEndian:
    def test_little_endian(self, isa):
        result = prepare_basis(isa, [0, 1], bitstring="10", endianness="little")
        assert ops(result) == [("x", [0])]

    def test_big_endian(self, isa):
        result = prepare_basis(isa, [0, 1], bitstring="10", endianness="big")
        assert ops(result) == [("x", [1])]


# ===========================================================================
# prepare_basis — validation errors
# ===========================================================================

class TestPrepareBasisErrors:
    def test_length_mismatch(self, isa):
        with pytest.raises(ValueError, match="length"):
            prepare_basis(isa, [0, 1], bitstring="100")

    def test_invalid_chars(self, isa):
        with pytest.raises(ValueError, match="'0' and '1'"):
            prepare_basis(isa, [0, 1], bitstring="0x")

    def test_invalid_basis(self, isa):
        with pytest.raises(ValueError, match="Basis"):
            prepare_basis(isa, [0], bitstring="0", basis="W")

    def test_missing_bitstring(self, isa):
        with pytest.raises(KeyError):
            prepare_basis(isa, [0])


# ===========================================================================
# prepare_uniform
# ===========================================================================

class TestPrepareUniform:
    def test_all_qubits(self, isa):
        result = prepare_uniform(isa, [0, 1, 2, 3])
        assert ops(result) == [("h", [0]), ("h", [1]), ("h", [2]), ("h", [3])]

    def test_subset(self, isa):
        result = prepare_uniform(isa, [0, 1, 2, 3], qubits=[1, 3])
        assert ops(result) == [("h", [1]), ("h", [3])]

    def test_single_qubit(self, isa):
        result = prepare_uniform(isa, [0, 1, 2], qubits=[2])
        assert ops(result) == [("h", [2])]

    def test_empty_subset(self, isa):
        result = prepare_uniform(isa, [0, 1], qubits=[])
        assert result == []

    def test_invalid_subset_raises(self, isa):
        with pytest.raises(ValueError, match="subset"):
            prepare_uniform(isa, [0, 1], qubits=[0, 5])


# ===========================================================================
# prepare_state — basic cases
# ===========================================================================

class TestPrepareStateBasic:
    def test_ground_state_no_ops(self, isa):
        """Preparing |0⟩ from |0⟩ should produce no gates."""
        result = prepare_state(isa, [0], state=[1, 0])
        assert result == []

    def test_excited_state_single_ry(self, isa):
        """|1⟩ on one qubit → Ry(π)."""
        result = prepare_state(isa, [0], state=[0, 1])
        assert len(result) == 1
        assert result[0].symbol == "ry"
        assert result[0].params[0] == pytest.approx(np.pi)

    def test_plus_state_single_ry(self, isa):
        """|+⟩ = (|0⟩+|1⟩)/√2 → Ry(π/2)."""
        result = prepare_state(isa, [0], state=[1 / np.sqrt(2), 1 / np.sqrt(2)])
        assert len(result) == 1
        assert result[0].symbol == "ry"
        assert result[0].params[0] == pytest.approx(np.pi / 2)

    def test_auto_normalizes(self, isa):
        """Unnormalized [3, 4] should produce the same result as [0.6, 0.8]."""
        result = prepare_state(isa, [0], state=[3, 4])
        assert len(result) == 1
        assert result[0].symbol == "ry"
        assert result[0].params[0] == pytest.approx(2 * np.arctan2(0.8, 0.6))


# ===========================================================================
# prepare_state — two-qubit cases
# ===========================================================================

class TestPrepareStateTwoQubit:
    def test_basis_01(self, isa):
        """|01⟩ = [0,1,0,0] → Ry(π) on qubit 0 only."""
        result = prepare_state(isa, [0, 1], state=[0, 1, 0, 0])
        syms = symbols(result)
        # Only Ry gates (no CX needed for a basis state on one qubit)
        assert "cx" not in syms
        assert "ry" in syms

    def test_bell_state_uses_cx(self, isa):
        """Bell state (|00⟩+|11⟩)/√2 requires entangling CX."""
        bell = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        result = prepare_state(isa, [0, 1], state=bell)
        syms = symbols(result)
        assert "cx" in syms
        assert "ry" in syms

    def test_equal_superposition(self, isa):
        """Uniform state [0.5, 0.5, 0.5, 0.5] → Ry on both qubits."""
        state = [0.5, 0.5, 0.5, 0.5]
        result = prepare_state(isa, [0, 1], state=state)
        # Level 0: Ry(π/2) on qubit 0, Level 1: Ry(π/2) on qubit 1
        ry_instrs = [i for i in result if i.symbol == "ry"]
        assert len(ry_instrs) >= 2

    def test_ground_state_2q_no_ops(self, isa):
        """|00⟩ → no gates needed."""
        result = prepare_state(isa, [0, 1], state=[1, 0, 0, 0])
        assert result == []


# ===========================================================================
# prepare_state — three-qubit case
# ===========================================================================

class TestPrepareStateThreeQubit:
    def test_ghz_like_state(self, isa):
        """(|000⟩+|111⟩)/√2 should produce a multi-level decomposition."""
        ghz = [0] * 8
        ghz[0] = 1 / np.sqrt(2)
        ghz[7] = 1 / np.sqrt(2)
        result = prepare_state(isa, [0, 1, 2], state=ghz)
        syms = symbols(result)
        assert "ry" in syms
        assert "cx" in syms
        # GHZ needs entanglement at multiple levels
        assert len(result) > 3

    def test_basis_state_101(self, isa):
        """|101⟩ = index 5 → decomposition produces Ry rotations."""
        state = [0] * 8
        state[5] = 1.0
        result = prepare_state(isa, [0, 1, 2], state=state)
        syms = symbols(result)
        assert "ry" in syms
        assert len(result) > 0


# ===========================================================================
# prepare_state — endianness
# ===========================================================================

class TestPrepareStateEndian:
    def test_big_endian_reverses_targets(self, isa):
        """Big endian should produce different qubit assignments."""
        state = [0, 1, 0, 0]  # |01⟩
        little = prepare_state(isa, [0, 1], state=state, endianness="little")
        big = prepare_state(isa, [0, 1], state=state, endianness="big")

        little_tgts = [i.target_qubits for i in little]
        big_tgts = [i.target_qubits for i in big]
        assert little_tgts != big_tgts


# ===========================================================================
# prepare_state — validation errors
# ===========================================================================

class TestPrepareStateErrors:
    def test_wrong_length(self, isa):
        with pytest.raises(ValueError, match="length"):
            prepare_state(isa, [0, 1], state=[1, 0, 0])

    def test_zero_norm(self, isa):
        with pytest.raises(ValueError, match="zero norm"):
            prepare_state(isa, [0], state=[0, 0])

    def test_missing_state(self, isa):
        with pytest.raises(KeyError):
            prepare_state(isa, [0])


# ===========================================================================
# prepare_state — complex phases
# ===========================================================================

class TestPrepareStateComplex:
    def test_complex_state_uses_rz(self, isa):
        """A state with non-trivial phases should produce Rz gates."""
        state = [1 / np.sqrt(2), 1j / np.sqrt(2)]
        result = prepare_state(isa, [0], state=state)
        syms = symbols(result)
        assert "rz" in syms

    def test_real_state_no_rz(self, isa):
        """A purely real state should not need Rz gates."""
        state = [1 / np.sqrt(2), 1 / np.sqrt(2)]
        result = prepare_state(isa, [0], state=state)
        syms = symbols(result)
        assert "rz" not in syms

    def test_complex_two_qubit(self, isa):
        """Complex 2-qubit state produces both Ry and Rz."""
        state = [0.5, 0.5j, -0.5, 0.5j]
        result = prepare_state(isa, [0, 1], state=state)
        syms = symbols(result)
        assert "ry" in syms
        assert "rz" in syms


# ===========================================================================
# _ucr — uniformly controlled rotation helper
# ===========================================================================

class TestUCR:
    def test_no_controls_single_rotation(self, isa):
        result = _ucr(isa, "ry", 0, [], [np.pi / 4])
        assert len(result) == 1
        assert result[0].symbol == "ry"
        assert result[0].params[0] == pytest.approx(np.pi / 4)

    def test_zero_angle_skipped(self, isa):
        result = _ucr(isa, "ry", 0, [], [0.0])
        assert result == []

    def test_all_zero_angles_skipped(self, isa):
        result = _ucr(isa, "rz", 0, [1], [0.0, 0.0])
        assert result == []

    def test_one_control_structure(self, isa):
        """UCR with 1 control → pattern: Ry, CX, Ry, CX."""
        result = _ucr(isa, "ry", 1, [0], [np.pi / 2, np.pi / 4])
        syms = symbols(result)
        # The recursive decomposition produces: UCR(α) CX UCR(β) CX
        # With one base angle each side
        assert syms.count("cx") == 2
        assert "ry" in syms

    def test_two_controls_structure(self, isa):
        """UCR with 2 controls → 6 CX gates (2 per sub-UCR × 2 + 2 top-level)."""
        angles = [0.1, 0.2, 0.3, 0.4]
        result = _ucr(isa, "ry", 2, [0, 1], angles)
        syms = symbols(result)
        assert syms.count("cx") == 6

    def test_rz_gate_type(self, isa):
        """Verify Rz gate type is produced when requested."""
        result = _ucr(isa, "rz", 0, [], [np.pi / 3])
        assert len(result) == 1
        assert result[0].symbol == "rz"
        assert result[0].params[0] == pytest.approx(np.pi / 3)

    def test_control_qubit_identity(self, isa):
        """CX gates should reference the correct control qubits."""
        result = _ucr(isa, "ry", 2, [0, 1], [0.5, 0.3, 0.2, 0.1])
        cx_instrs = [i for i in result if i.symbol == "cx"]
        ctrl_qubits = {i.control_qubits[0] for i in cx_instrs}
        # Should use controls 0 and 1
        assert ctrl_qubits == {0, 1}

    def test_target_qubit_preserved(self, isa):
        """All gates (Ry and CX) should target the specified qubit."""
        result = _ucr(isa, "ry", 5, [3], [0.1, 0.2])
        for instr in result:
            assert 5 in instr.target_qubits


# ===========================================================================
# BlockFactory integration — dispatch still works
# ===========================================================================

class TestBlockFactoryIntegration:
    @pytest.fixture
    def factory(self):
        from lccfq_lang.backend import QPU
        from lccfq_lang.arch.register import CRegister
        from lccfq_lang.lang.blocks import BlockFactory
        qpu = QPU(filename="src/tests/data/testing.toml")
        qreg = qpu.qregister(4)
        creg = CRegister(4)
        return BlockFactory(qreg, creg)

    def test_dispatch_prepare_basis(self, factory):
        from lccfq_lang.lang.blocks import BlockType
        result = factory.block(BlockType.PREPAREBASIS, [0, 1], bitstring="10")
        assert len(result) == 1
        assert result[0].symbol == "x"

    def test_dispatch_prepare_uniform(self, factory):
        from lccfq_lang.lang.blocks import BlockType
        result = factory.block(BlockType.PREPAREUNIFORM, [0, 1, 2])
        assert len(result) == 3
        assert all(i.symbol == "h" for i in result)

    def test_dispatch_prepare_state(self, factory):
        from lccfq_lang.lang.blocks import BlockType
        result = factory.block(
            BlockType.PREPARESTATE, [0, 1],
            state=[0, 1, 0, 0]
        )
        assert len(result) > 0
