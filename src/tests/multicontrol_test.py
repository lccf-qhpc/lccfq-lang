"""
Filename: multicontrol_test.py
Author: Santiago Nunez-Corrales
Date: 2026-05-15
Version: 1.0
Description:
    Comprehensive tests for multi-controlled gate decompositions:
    MCX, MCZ, MCRY, MCRZ.

    Coverage:
      - Input validation (TypeError / ValueError for every erroneous input)
      - Edge cases n=0,1 for both modes
      - Semantic equivalence (unitary matrix comparison) for n=2..6 / n=2..5
      - Structural invariant: no emitted Instruction has >1 control qubit
      - Ancilla restoration for barenco mode
      - BlockFactory dispatch round-trip

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import math
import numpy as np
import pytest
from pathlib import Path

from lccfq_lang.arch.isa import ISA
from lccfq_lang.lang.multicontrol import mcx, mcz, mcry, mcrz
from tests._sim import simulate, basis_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def isa():
    return ISA("test")


DATA_DIR = Path(__file__).parent / "data"
CONFIG = str(DATA_DIR / "testing.toml")


@pytest.fixture
def factory():
    from lccfq_lang.backend import QPU
    from lccfq_lang.arch.register import CRegister
    from lccfq_lang.lang.blocks import BlockFactory
    qpu = QPU(filename=CONFIG)
    qreg = qpu.qregister(8)
    creg = CRegister(0)
    return BlockFactory(qreg, creg)


# ---------------------------------------------------------------------------
# Reference matrix helpers
# ---------------------------------------------------------------------------

def _reference_mcx_matrix(n_ctrl: int) -> np.ndarray:
    """Exact unitary for C^n(X) in little-endian convention.

    Qubit layout: control[0] = bit 1, control[1] = bit 2, ..., target = bit 0.
    The gate flips target (bit 0) iff all control bits are 1.
    """
    n = n_ctrl + 1
    dim = 1 << n
    M = np.eye(dim, dtype=complex)
    # ctrl_mask: bits for control qubits (bits 1..n_ctrl, since target=bit 0)
    ctrl_mask = ((1 << n) - 1) ^ 1   # all bits except bit 0
    for idx in range(dim):
        if (idx & ctrl_mask) == ctrl_mask:
            j = idx ^ 1             # flip bit 0 (target)
            M[idx, idx] = 0.0
            M[j, idx] = 1.0
    return M


def _reference_mcz_matrix(n_ctrl: int) -> np.ndarray:
    """Exact unitary for C^n(Z) in little-endian convention.

    Flips the phase of the all-ones state.
    """
    n = n_ctrl + 1
    dim = 1 << n
    M = np.eye(dim, dtype=complex)
    # The all-ones state index
    M[dim - 1, dim - 1] = -1.0
    return M


def _reference_mcry_matrix(n_ctrl: int, theta: float) -> np.ndarray:
    """Exact unitary for C^n(Ry(theta)) in little-endian convention."""
    n = n_ctrl + 1
    dim = 1 << n
    M = np.eye(dim, dtype=complex)
    ctrl_mask = ((1 << n) - 1) ^ 1
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    Ry = np.array([[c, -s], [s, c]], dtype=complex)
    for idx in range(dim):
        if (idx & ctrl_mask) == ctrl_mask:
            j = idx ^ 1
            if idx < j:   # process each (idx,j) pair once
                # |0> branch: idx has bit0=0, j has bit0=1
                a, b = M[idx, idx], M[j, idx]
                M[idx, idx] = Ry[0, 0] * a + Ry[0, 1] * b
                M[j, idx] = Ry[1, 0] * a + Ry[1, 1] * b
                a2, b2 = M[idx, j], M[j, j]
                M[idx, j] = Ry[0, 0] * a2 + Ry[0, 1] * b2
                M[j, j] = Ry[1, 0] * a2 + Ry[1, 1] * b2
    return M


def _reference_mcrz_matrix(n_ctrl: int, theta: float) -> np.ndarray:
    """Exact unitary for C^n(Rz(theta)) in little-endian convention."""
    n = n_ctrl + 1
    dim = 1 << n
    M = np.eye(dim, dtype=complex)
    ctrl_mask = ((1 << n) - 1) ^ 1
    e_minus = math.e ** (-1j * theta / 2)
    e_plus = math.e ** (1j * theta / 2)
    for idx in range(dim):
        if (idx & ctrl_mask) == ctrl_mask:
            bit0 = (idx >> 0) & 1
            M[idx, idx] = e_minus if bit0 == 0 else e_plus
    return M


def _simulate_unitary(instructions, n: int) -> np.ndarray:
    """Build the full unitary by simulating each basis state as an input."""
    dim = 1 << n
    cols = []
    for col in range(dim):
        col_out = simulate(instructions, n, initial=basis_state(col, n))
        cols.append(col_out)
    return np.column_stack(cols)


def _assert_unitary_close(U_got: np.ndarray, U_ref: np.ndarray, atol=1e-10,
                           label: str = ""):
    """Assert two unitaries agree up to a global phase."""
    # Find a non-zero element to extract global phase
    nonzero = np.argmax(np.abs(U_ref))
    row, col = np.unravel_index(nonzero, U_ref.shape)
    if abs(U_got[row, col]) < 1e-14:
        # Both matrices should be near-zero here — just check Frobenius
        err = np.linalg.norm(U_got - U_ref)
        assert err < atol, f"{label}: Frobenius error {err:.3e} >= {atol}"
        return
    phase = U_got[row, col] / U_ref[row, col]
    phase /= abs(phase)
    err = np.linalg.norm(U_got - phase * U_ref)
    assert err < atol, f"{label}: Frobenius error (after phase) {err:.3e} >= {atol}"


def _no_multi_ctrl(instructions) -> bool:
    """Return True iff every instruction has at most 1 control qubit."""
    return all(
        inst.control_qubits is None or len(inst.control_qubits) <= 1
        for inst in instructions
    )


# ===========================================================================
# TestMcxValidation — input validation for mcx
# ===========================================================================

class TestMcxValidation:
    def test_missing_tg_raises_typeerror(self, isa):
        with pytest.raises(TypeError, match="tg"):
            mcx(isa, [0, 1], mode="vchain")

    def test_missing_mode_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="mode"):
            mcx(isa, [0, 1], tg=2)

    def test_invalid_mode_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="mode"):
            mcx(isa, [0, 1], tg=2, mode="auto")

    def test_tg_in_controls_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="target"):
            mcx(isa, [0, 1], tg=0, mode="vchain")

    def test_duplicate_control_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="duplicate"):
            mcx(isa, [0, 0, 1], tg=2, mode="vchain")

    def test_barenco_no_ancilla_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="ancilla"):
            mcx(isa, [0, 1], tg=2, mode="barenco")

    def test_ancilla_overlaps_target_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="ancilla"):
            mcx(isa, [0, 1], tg=2, mode="barenco", ancilla=2)

    def test_ancilla_overlaps_controls_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="ancilla"):
            mcx(isa, [0, 1], tg=2, mode="barenco", ancilla=0)


# ===========================================================================
# TestMczValidation — input validation for mcz
# ===========================================================================

class TestMczValidation:
    def test_missing_tg_raises_typeerror(self, isa):
        with pytest.raises(TypeError, match="tg"):
            mcz(isa, [0, 1], mode="vchain")

    def test_missing_mode_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="mode"):
            mcz(isa, [0, 1], tg=2)

    def test_invalid_mode_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="mode"):
            mcz(isa, [0, 1], tg=2, mode="direct")

    def test_tg_in_controls_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="target"):
            mcz(isa, [0, 1], tg=1, mode="vchain")

    def test_duplicate_control_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="duplicate"):
            mcz(isa, [0, 0], tg=2, mode="vchain")

    def test_barenco_no_ancilla_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="ancilla"):
            mcz(isa, [0, 1], tg=2, mode="barenco")

    def test_ancilla_overlaps_target_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="ancilla"):
            mcz(isa, [0, 1], tg=2, mode="barenco", ancilla=2)

    def test_ancilla_overlaps_controls_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="ancilla"):
            mcz(isa, [0, 1], tg=2, mode="barenco", ancilla=1)


# ===========================================================================
# TestMcryValidation — input validation for mcry
# ===========================================================================

class TestMcryValidation:
    def test_missing_tg_raises_typeerror(self, isa):
        with pytest.raises(TypeError, match="tg"):
            mcry(isa, [0, 1], mode="vchain", theta=1.0)

    def test_missing_theta_raises_typeerror(self, isa):
        with pytest.raises(TypeError, match="theta"):
            mcry(isa, [0, 1], tg=2, mode="vchain")

    def test_invalid_theta_raises_typeerror(self, isa):
        with pytest.raises(TypeError, match="theta"):
            mcry(isa, [0, 1], tg=2, mode="vchain", theta="pi")

    def test_missing_mode_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="mode"):
            mcry(isa, [0, 1], tg=2, theta=1.0)

    def test_invalid_mode_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="mode"):
            mcry(isa, [0, 1], tg=2, theta=1.0, mode="bad")

    def test_tg_in_controls_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="target"):
            mcry(isa, [0, 1], tg=0, mode="vchain", theta=1.0)

    def test_barenco_no_ancilla_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="ancilla"):
            mcry(isa, [0, 1], tg=2, mode="barenco", theta=1.0)

    def test_ancilla_overlaps_controls_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="ancilla"):
            mcry(isa, [0, 1], tg=2, mode="barenco", theta=1.0, ancilla=0)


# ===========================================================================
# TestMcrzValidation — input validation for mcrz
# ===========================================================================

class TestMcrzValidation:
    def test_missing_tg_raises_typeerror(self, isa):
        with pytest.raises(TypeError, match="tg"):
            mcrz(isa, [0, 1], mode="vchain", theta=1.0)

    def test_missing_theta_raises_typeerror(self, isa):
        with pytest.raises(TypeError, match="theta"):
            mcrz(isa, [0, 1], tg=2, mode="vchain")

    def test_invalid_theta_raises_typeerror(self, isa):
        with pytest.raises(TypeError, match="theta"):
            mcrz(isa, [0, 1], tg=2, mode="vchain", theta=1+2j)

    def test_missing_mode_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="mode"):
            mcrz(isa, [0, 1], tg=2, theta=1.0)

    def test_invalid_mode_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="mode"):
            mcrz(isa, [0, 1], tg=2, theta=1.0, mode="recursive")

    def test_tg_in_controls_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="target"):
            mcrz(isa, [0, 2], tg=2, mode="vchain", theta=1.0)

    def test_barenco_no_ancilla_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="ancilla"):
            mcrz(isa, [0, 1], tg=2, mode="barenco", theta=1.0)

    def test_ancilla_overlaps_target_raises_valueerror(self, isa):
        with pytest.raises(ValueError, match="ancilla"):
            mcrz(isa, [0, 1], tg=2, mode="barenco", theta=1.0, ancilla=2)


# ===========================================================================
# TestMcxEdgeCases — n=0,1 for both modes
# ===========================================================================

class TestMcxEdgeCases:
    def test_n0_emits_x_vchain(self, isa):
        result = mcx(isa, [], tg=0, mode="vchain")
        assert len(result) == 1
        assert result[0].symbol == "x"
        assert result[0].target_qubits == [0]
        assert not result[0].is_controlled

    def test_n0_emits_x_barenco(self, isa):
        result = mcx(isa, [], tg=0, mode="barenco")
        assert len(result) == 1
        assert result[0].symbol == "x"

    def test_n1_emits_cx_vchain(self, isa):
        result = mcx(isa, [3], tg=5, mode="vchain")
        assert len(result) == 1
        assert result[0].symbol == "cx"
        assert result[0].control_qubits == [3]
        assert result[0].target_qubits == [5]

    def test_n1_emits_cx_barenco(self, isa):
        # barenco with n=1 does not need ancilla (validation allows None)
        result = mcx(isa, [3], tg=5, mode="barenco")
        assert len(result) == 1
        assert result[0].symbol == "cx"

    def test_n2_vchain_no_ancilla(self, isa):
        result = mcx(isa, [0, 1], tg=2, mode="vchain")
        assert _no_multi_ctrl(result)

    def test_n2_barenco_with_ancilla(self, isa):
        result = mcx(isa, [0, 1], tg=2, mode="barenco", ancilla=3)
        assert _no_multi_ctrl(result)


# ===========================================================================
# TestMczEdgeCases — n=0,1,2 for both modes
# ===========================================================================

class TestMczEdgeCases:
    def test_n0_emits_z_vchain(self, isa):
        result = mcz(isa, [], tg=0, mode="vchain")
        assert len(result) == 1
        assert result[0].symbol == "z"
        assert not result[0].is_controlled

    def test_n0_emits_z_barenco(self, isa):
        result = mcz(isa, [], tg=1, mode="barenco")
        assert len(result) == 1
        assert result[0].symbol == "z"

    def test_n1_emits_cz_vchain(self, isa):
        result = mcz(isa, [2], tg=4, mode="vchain")
        assert len(result) == 1
        assert result[0].symbol == "cz"
        assert result[0].control_qubits == [2]
        assert result[0].target_qubits == [4]

    def test_n1_emits_cz_barenco(self, isa):
        result = mcz(isa, [2], tg=4, mode="barenco")
        assert len(result) == 1
        assert result[0].symbol == "cz"

    def test_n2_vchain_structural(self, isa):
        result = mcz(isa, [0, 1], tg=2, mode="vchain")
        # MCZ = H . MCX . H on target; must have H as first and last
        assert result[0].symbol == "h" and result[0].target_qubits == [2]
        assert result[-1].symbol == "h" and result[-1].target_qubits == [2]
        assert _no_multi_ctrl(result)

    def test_n2_barenco_with_ancilla(self, isa):
        result = mcz(isa, [0, 1], tg=2, mode="barenco", ancilla=3)
        assert _no_multi_ctrl(result)


# ===========================================================================
# TestMcryEdgeCases — n=0,1 for both modes
# ===========================================================================

class TestMcryEdgeCases:
    def test_n0_emits_ry_vchain(self, isa):
        result = mcry(isa, [], tg=0, mode="vchain", theta=math.pi / 3)
        assert len(result) == 1
        assert result[0].symbol == "ry"
        assert result[0].params[0] == pytest.approx(math.pi / 3)

    def test_n0_emits_ry_barenco(self, isa):
        result = mcry(isa, [], tg=0, mode="barenco", theta=math.pi / 4)
        assert len(result) == 1
        assert result[0].symbol == "ry"

    def test_n0_theta_zero_identity(self, isa):
        result = mcry(isa, [], tg=0, mode="vchain", theta=0.0)
        assert len(result) == 1
        assert result[0].params[0] == pytest.approx(0.0)

    def test_n1_emits_cry_vchain(self, isa):
        result = mcry(isa, [1], tg=3, mode="vchain", theta=math.pi / 2)
        assert len(result) == 1
        assert result[0].symbol == "cry"
        assert result[0].control_qubits == [1]
        assert result[0].target_qubits == [3]

    def test_n1_emits_cry_barenco(self, isa):
        result = mcry(isa, [1], tg=3, mode="barenco", theta=1.5)
        assert len(result) == 1
        assert result[0].symbol == "cry"

    def test_n2_vchain_no_ancilla(self, isa):
        result = mcry(isa, [0, 1], tg=2, mode="vchain", theta=math.pi)
        assert _no_multi_ctrl(result)

    def test_n2_barenco_with_ancilla(self, isa):
        result = mcry(isa, [0, 1], tg=2, mode="barenco", theta=math.pi, ancilla=3)
        assert _no_multi_ctrl(result)


# ===========================================================================
# TestMcrzEdgeCases — n=0,1 for both modes
# ===========================================================================

class TestMcrzEdgeCases:
    def test_n0_emits_rz_vchain(self, isa):
        result = mcrz(isa, [], tg=0, mode="vchain", theta=math.pi / 2)
        assert len(result) == 1
        assert result[0].symbol == "rz"
        assert result[0].params[0] == pytest.approx(math.pi / 2)

    def test_n0_emits_rz_barenco(self, isa):
        result = mcrz(isa, [], tg=0, mode="barenco", theta=0.7)
        assert len(result) == 1
        assert result[0].symbol == "rz"

    def test_n1_emits_crz_vchain(self, isa):
        result = mcrz(isa, [2], tg=4, mode="vchain", theta=1.2)
        assert len(result) == 1
        assert result[0].symbol == "crz"
        assert result[0].control_qubits == [2]
        assert result[0].target_qubits == [4]

    def test_n1_emits_crz_barenco(self, isa):
        result = mcrz(isa, [2], tg=4, mode="barenco", theta=-1.0)
        assert len(result) == 1
        assert result[0].symbol == "crz"

    def test_n2_vchain_no_ancilla(self, isa):
        result = mcrz(isa, [0, 1], tg=2, mode="vchain", theta=math.pi / 3)
        assert _no_multi_ctrl(result)

    def test_n2_barenco_with_ancilla(self, isa):
        result = mcrz(isa, [0, 1], tg=2, mode="barenco", theta=1.0, ancilla=3)
        assert _no_multi_ctrl(result)


# ===========================================================================
# TestMcxSemantic — unitary correctness for n=2..6, both modes
# ===========================================================================

_MCX_N_RANGE = [2, 3, 4, 5, 6]
_MCX_MODES = ["vchain", "barenco"]


@pytest.mark.parametrize("n_ctrl", _MCX_N_RANGE)
@pytest.mark.parametrize("mode", _MCX_MODES)
class TestMcxSemantic:
    def test_unitary_matches_reference(self, isa, n_ctrl, mode):
        """Simulated MCX unitary must match the reference matrix up to global phase."""
        # qubits: controls = [1..n_ctrl], target = 0, ancilla = n_ctrl+1 if needed
        controls = list(range(1, n_ctrl + 1))
        target = 0
        n_total = n_ctrl + 1
        ancilla_q = n_ctrl + 1  # extra qubit used only in barenco mode

        if mode == "barenco":
            instrs = mcx(isa, controls, tg=target, mode=mode, ancilla=ancilla_q)
            n_sim = n_ctrl + 2   # include ancilla qubit
        else:
            instrs = mcx(isa, controls, tg=target, mode=mode)
            n_sim = n_total

        assert _no_multi_ctrl(instrs), f"MCX n={n_ctrl} mode={mode}: multi-ctrl gate emitted"

        if mode == "vchain":
            # For vchain, simulate over n_total qubits and compare to reference
            U_got = _simulate_unitary(instrs, n_sim)
            U_ref = _reference_mcx_matrix(n_ctrl)
            _assert_unitary_close(U_got, U_ref, atol=1e-9,
                                  label=f"MCX n={n_ctrl} mode={mode}")
        else:
            # For barenco, the ancilla starts and ends in |0>; verify semantics
            # on just the computational (non-ancilla) qubits by checking each
            # basis-state column of the MCX unitary on the n_total qubits.
            dim_comp = 1 << n_total
            U_ref = _reference_mcx_matrix(n_ctrl)
            for col in range(dim_comp):
                # Input: computational basis state on [target] + controls, ancilla=|0>
                init = np.zeros(1 << n_sim, dtype=complex)
                init[col] = 1.0   # ancilla bit is the highest bit -> stays 0
                out = simulate(instrs, n_sim, initial=init)
                # Marginalise: project ancilla back to |0> (check ancilla is restored)
                anc_bit = n_sim - 1  # ancilla qubit index in the simulator
                # Sum over ancilla=1 amplitudes — should be ~0
                anc_mask = 1 << anc_bit
                amp_anc1 = sum(abs(out[idx]) ** 2 for idx in range(1 << n_sim)
                               if (idx & anc_mask) != 0)
                assert amp_anc1 < 1e-10, \
                    f"MCX barenco n={n_ctrl}: ancilla not restored for col {col}"
                # Extract the computational part (ancilla=0 subspace)
                out_comp = np.array(
                    [out[idx] for idx in range(1 << n_sim) if (idx & anc_mask) == 0],
                    dtype=complex
                )
                expected_col = U_ref[:, col]
                assert np.allclose(out_comp, expected_col, atol=1e-9), \
                    f"MCX barenco n={n_ctrl}: wrong output for col {col}"


# ===========================================================================
# TestMczSemantic — unitary correctness for n=2..6, both modes
# ===========================================================================

_MCZ_N_RANGE = [2, 3, 4, 5, 6]
_MCZ_MODES = ["vchain", "barenco"]


@pytest.mark.parametrize("n_ctrl", _MCZ_N_RANGE)
@pytest.mark.parametrize("mode", _MCZ_MODES)
class TestMczSemantic:
    def test_unitary_matches_reference(self, isa, n_ctrl, mode):
        """Simulated MCZ unitary must match the reference matrix."""
        controls = list(range(1, n_ctrl + 1))
        target = 0
        n_total = n_ctrl + 1
        ancilla_q = n_ctrl + 1

        if mode == "barenco":
            instrs = mcz(isa, controls, tg=target, mode=mode, ancilla=ancilla_q)
            n_sim = n_ctrl + 2
        else:
            instrs = mcz(isa, controls, tg=target, mode=mode)
            n_sim = n_total

        assert _no_multi_ctrl(instrs), f"MCZ n={n_ctrl} mode={mode}: multi-ctrl gate emitted"

        if mode == "vchain":
            U_got = _simulate_unitary(instrs, n_sim)
            U_ref = _reference_mcz_matrix(n_ctrl)
            _assert_unitary_close(U_got, U_ref, atol=1e-9,
                                  label=f"MCZ n={n_ctrl} mode={mode}")
        else:
            dim_comp = 1 << n_total
            U_ref = _reference_mcz_matrix(n_ctrl)
            for col in range(dim_comp):
                init = np.zeros(1 << n_sim, dtype=complex)
                init[col] = 1.0
                out = simulate(instrs, n_sim, initial=init)
                anc_bit = n_sim - 1
                anc_mask = 1 << anc_bit
                amp_anc1 = sum(abs(out[idx]) ** 2 for idx in range(1 << n_sim)
                               if (idx & anc_mask) != 0)
                assert amp_anc1 < 1e-10, \
                    f"MCZ barenco n={n_ctrl}: ancilla not restored for col {col}"
                out_comp = np.array(
                    [out[idx] for idx in range(1 << n_sim) if (idx & anc_mask) == 0],
                    dtype=complex
                )
                expected_col = U_ref[:, col]
                assert np.allclose(out_comp, expected_col, atol=1e-9), \
                    f"MCZ barenco n={n_ctrl}: wrong output for col {col}"


# ===========================================================================
# TestMcrySemantic — unitary correctness for n=2..5, angle sweep, both modes
# ===========================================================================

_MCRY_N_RANGE = [2, 3, 4, 5]
_MCRY_ANGLES = [0.0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 2,
                -math.pi / 3, 2.7]
_MCRY_MODES = ["vchain", "barenco"]


@pytest.mark.parametrize("theta", _MCRY_ANGLES)
@pytest.mark.parametrize("n_ctrl", _MCRY_N_RANGE)
@pytest.mark.parametrize("mode", _MCRY_MODES)
class TestMcrySemantic:
    def test_unitary_matches_reference(self, isa, n_ctrl, theta, mode):
        """Simulated MCRY unitary must match the reference matrix."""
        controls = list(range(1, n_ctrl + 1))
        target = 0
        n_total = n_ctrl + 1
        ancilla_q = n_ctrl + 1

        if mode == "barenco":
            instrs = mcry(isa, controls, tg=target, mode=mode, theta=theta,
                          ancilla=ancilla_q)
            n_sim = n_ctrl + 2
        else:
            instrs = mcry(isa, controls, tg=target, mode=mode, theta=theta)
            n_sim = n_total

        assert _no_multi_ctrl(instrs), \
            f"MCRY n={n_ctrl} theta={theta:.3f} mode={mode}: multi-ctrl gate emitted"

        if mode == "vchain":
            U_got = _simulate_unitary(instrs, n_sim)
            U_ref = _reference_mcry_matrix(n_ctrl, theta)
            _assert_unitary_close(U_got, U_ref, atol=1e-9,
                                  label=f"MCRY n={n_ctrl} theta={theta:.3f} mode={mode}")
        else:
            dim_comp = 1 << n_total
            U_ref = _reference_mcry_matrix(n_ctrl, theta)
            for col in range(dim_comp):
                init = np.zeros(1 << n_sim, dtype=complex)
                init[col] = 1.0
                out = simulate(instrs, n_sim, initial=init)
                anc_bit = n_sim - 1
                anc_mask = 1 << anc_bit
                amp_anc1 = sum(abs(out[idx]) ** 2 for idx in range(1 << n_sim)
                               if (idx & anc_mask) != 0)
                assert amp_anc1 < 1e-10, \
                    f"MCRY barenco n={n_ctrl} theta={theta:.3f}: ancilla not restored for col {col}"
                out_comp = np.array(
                    [out[idx] for idx in range(1 << n_sim) if (idx & anc_mask) == 0],
                    dtype=complex
                )
                expected_col = U_ref[:, col]
                assert np.allclose(out_comp, expected_col, atol=1e-9), \
                    f"MCRY barenco n={n_ctrl} theta={theta:.3f}: wrong output for col {col}"


# ===========================================================================
# TestMcrzSemantic — unitary correctness for n=2..5, angle sweep, both modes
# ===========================================================================

_MCRZ_N_RANGE = [2, 3, 4, 5]
_MCRZ_ANGLES = [0.0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 2,
                -math.pi / 3, 2.7]
_MCRZ_MODES = ["vchain", "barenco"]


@pytest.mark.parametrize("theta", _MCRZ_ANGLES)
@pytest.mark.parametrize("n_ctrl", _MCRZ_N_RANGE)
@pytest.mark.parametrize("mode", _MCRZ_MODES)
class TestMcrzSemantic:
    def test_unitary_matches_reference(self, isa, n_ctrl, theta, mode):
        """Simulated MCRZ unitary must match the reference matrix."""
        controls = list(range(1, n_ctrl + 1))
        target = 0
        n_total = n_ctrl + 1
        ancilla_q = n_ctrl + 1

        if mode == "barenco":
            instrs = mcrz(isa, controls, tg=target, mode=mode, theta=theta,
                          ancilla=ancilla_q)
            n_sim = n_ctrl + 2
        else:
            instrs = mcrz(isa, controls, tg=target, mode=mode, theta=theta)
            n_sim = n_total

        assert _no_multi_ctrl(instrs), \
            f"MCRZ n={n_ctrl} theta={theta:.3f} mode={mode}: multi-ctrl gate emitted"

        if mode == "vchain":
            U_got = _simulate_unitary(instrs, n_sim)
            U_ref = _reference_mcrz_matrix(n_ctrl, theta)
            _assert_unitary_close(U_got, U_ref, atol=1e-9,
                                  label=f"MCRZ n={n_ctrl} theta={theta:.3f} mode={mode}")
        else:
            dim_comp = 1 << n_total
            U_ref = _reference_mcrz_matrix(n_ctrl, theta)
            for col in range(dim_comp):
                init = np.zeros(1 << n_sim, dtype=complex)
                init[col] = 1.0
                out = simulate(instrs, n_sim, initial=init)
                anc_bit = n_sim - 1
                anc_mask = 1 << anc_bit
                amp_anc1 = sum(abs(out[idx]) ** 2 for idx in range(1 << n_sim)
                               if (idx & anc_mask) != 0)
                assert amp_anc1 < 1e-10, \
                    f"MCRZ barenco n={n_ctrl} theta={theta:.3f}: ancilla not restored for col {col}"
                out_comp = np.array(
                    [out[idx] for idx in range(1 << n_sim) if (idx & anc_mask) == 0],
                    dtype=complex
                )
                expected_col = U_ref[:, col]
                assert np.allclose(out_comp, expected_col, atol=1e-9), \
                    f"MCRZ barenco n={n_ctrl} theta={theta:.3f}: wrong output for col {col}"


# ===========================================================================
# TestMcInvariant — no emitted gate has >1 control qubit
# ===========================================================================

class TestMcInvariant:
    """For every supported gate and n, every emitted Instruction has
    at most one control qubit — i.e., the decomposition is fully native."""

    @pytest.mark.parametrize("n_ctrl", [0, 1, 2, 3, 4, 5, 6])
    def test_mcx_vchain_invariant(self, isa, n_ctrl):
        controls = list(range(n_ctrl))
        result = mcx(isa, controls, tg=n_ctrl, mode="vchain")
        assert _no_multi_ctrl(result), \
            f"MCX vchain n={n_ctrl}: multi-ctrl instruction found"

    @pytest.mark.parametrize("n_ctrl", [0, 1, 2, 3, 4, 5, 6])
    def test_mcx_barenco_invariant(self, isa, n_ctrl):
        controls = list(range(n_ctrl))
        ancilla = n_ctrl + 1 if n_ctrl >= 2 else None
        result = mcx(isa, controls, tg=n_ctrl, mode="barenco", ancilla=ancilla)
        assert _no_multi_ctrl(result), \
            f"MCX barenco n={n_ctrl}: multi-ctrl instruction found"

    @pytest.mark.parametrize("n_ctrl", [0, 1, 2, 3, 4, 5, 6])
    def test_mcz_vchain_invariant(self, isa, n_ctrl):
        controls = list(range(n_ctrl))
        result = mcz(isa, controls, tg=n_ctrl, mode="vchain")
        assert _no_multi_ctrl(result), \
            f"MCZ vchain n={n_ctrl}: multi-ctrl instruction found"

    @pytest.mark.parametrize("n_ctrl", [0, 1, 2, 3, 4, 5, 6])
    def test_mcz_barenco_invariant(self, isa, n_ctrl):
        controls = list(range(n_ctrl))
        ancilla = n_ctrl + 1 if n_ctrl >= 2 else None
        result = mcz(isa, controls, tg=n_ctrl, mode="barenco", ancilla=ancilla)
        assert _no_multi_ctrl(result), \
            f"MCZ barenco n={n_ctrl}: multi-ctrl instruction found"

    @pytest.mark.parametrize("n_ctrl", [0, 1, 2, 3, 4, 5])
    def test_mcry_vchain_invariant(self, isa, n_ctrl):
        controls = list(range(n_ctrl))
        result = mcry(isa, controls, tg=n_ctrl, mode="vchain", theta=1.0)
        assert _no_multi_ctrl(result), \
            f"MCRY vchain n={n_ctrl}: multi-ctrl instruction found"

    @pytest.mark.parametrize("n_ctrl", [0, 1, 2, 3, 4, 5])
    def test_mcrz_vchain_invariant(self, isa, n_ctrl):
        controls = list(range(n_ctrl))
        result = mcrz(isa, controls, tg=n_ctrl, mode="vchain", theta=1.0)
        assert _no_multi_ctrl(result), \
            f"MCRZ vchain n={n_ctrl}: multi-ctrl instruction found"


# ===========================================================================
# TestMcAncillaRestoration — barenco mode restores ancilla to |0>
# ===========================================================================

class TestMcAncillaRestoration:
    """After executing a barenco-mode MCX or MCZ, the ancilla qubit returns
    to |0> for every computational basis-state input.  This is the core
    correctness promise of the Barenco clean-ancilla scheme."""

    @pytest.mark.parametrize("n_ctrl", [2, 3, 4, 5])
    def test_mcx_barenco_ancilla_restored(self, isa, n_ctrl):
        controls = list(range(1, n_ctrl + 1))
        target = 0
        ancilla_q = n_ctrl + 1
        instrs = mcx(isa, controls, tg=target, mode="barenco", ancilla=ancilla_q)
        n_sim = n_ctrl + 2  # total qubits including ancilla
        dim_comp = 1 << (n_ctrl + 1)
        anc_mask = 1 << ancilla_q

        for col in range(dim_comp):
            init = np.zeros(1 << n_sim, dtype=complex)
            init[col] = 1.0
            out = simulate(instrs, n_sim, initial=init)
            amp_anc1 = sum(abs(out[idx]) ** 2 for idx in range(1 << n_sim)
                           if (idx & anc_mask) != 0)
            assert amp_anc1 < 1e-10, \
                f"MCX barenco n={n_ctrl}: ancilla not restored for basis state {col}"

    @pytest.mark.parametrize("n_ctrl", [2, 3, 4])
    def test_mcz_barenco_ancilla_restored(self, isa, n_ctrl):
        controls = list(range(1, n_ctrl + 1))
        target = 0
        ancilla_q = n_ctrl + 1
        instrs = mcz(isa, controls, tg=target, mode="barenco", ancilla=ancilla_q)
        n_sim = n_ctrl + 2
        dim_comp = 1 << (n_ctrl + 1)
        anc_mask = 1 << ancilla_q

        for col in range(dim_comp):
            init = np.zeros(1 << n_sim, dtype=complex)
            init[col] = 1.0
            out = simulate(instrs, n_sim, initial=init)
            amp_anc1 = sum(abs(out[idx]) ** 2 for idx in range(1 << n_sim)
                           if (idx & anc_mask) != 0)
            assert amp_anc1 < 1e-10, \
                f"MCZ barenco n={n_ctrl}: ancilla not restored for basis state {col}"


# ===========================================================================
# TestMcBlockFactoryDispatch — factory.block() round-trips
# ===========================================================================

class TestMcBlockFactoryDispatch:
    def test_factory_mcx_vchain(self, factory):
        from lccfq_lang.lang.blocks import BlockType
        result = factory.block(BlockType.MCX, [0, 1, 2], tg=3, mode="vchain")
        assert _no_multi_ctrl(result)
        assert len(result) > 0

    def test_factory_mcx_barenco(self, factory):
        from lccfq_lang.lang.blocks import BlockType
        result = factory.block(BlockType.MCX, [0, 1, 2], tg=3, mode="barenco",
                               ancilla=4)
        assert _no_multi_ctrl(result)
        assert len(result) > 0

    def test_factory_mcz_vchain(self, factory):
        from lccfq_lang.lang.blocks import BlockType
        result = factory.block(BlockType.MCZ, [0, 1, 2], tg=3, mode="vchain")
        assert _no_multi_ctrl(result)
        # MCZ = H.MCX.H so first and last must be H on target
        assert result[0].symbol == "h"
        assert result[-1].symbol == "h"

    def test_factory_mcz_barenco(self, factory):
        from lccfq_lang.lang.blocks import BlockType
        result = factory.block(BlockType.MCZ, [0, 1, 2], tg=3, mode="barenco",
                               ancilla=4)
        assert _no_multi_ctrl(result)

    def test_factory_mcry_vchain(self, factory):
        from lccfq_lang.lang.blocks import BlockType
        result = factory.block(BlockType.MCRY, [0, 1], tg=2, mode="vchain",
                               theta=math.pi / 2)
        assert _no_multi_ctrl(result)
        # Pattern: Ry(theta/2) MCX Ry(-theta/2) MCX — starts with ry on target
        assert result[0].symbol == "ry"
        assert result[0].target_qubits == [2]

    def test_factory_mcrz_vchain(self, factory):
        from lccfq_lang.lang.blocks import BlockType
        result = factory.block(BlockType.MCRZ, [0, 1], tg=2, mode="vchain",
                               theta=math.pi / 3)
        assert _no_multi_ctrl(result)
        assert result[0].symbol == "rz"
        assert result[0].target_qubits == [2]

    def test_factory_mcx_n1(self, factory):
        from lccfq_lang.lang.blocks import BlockType
        result = factory.block(BlockType.MCX, [0], tg=1, mode="vchain")
        assert len(result) == 1
        assert result[0].symbol == "cx"

    def test_factory_mcz_n1(self, factory):
        from lccfq_lang.lang.blocks import BlockType
        result = factory.block(BlockType.MCZ, [0], tg=1, mode="vchain")
        assert len(result) == 1
        assert result[0].symbol == "cz"

    def test_factory_mcry_n1(self, factory):
        from lccfq_lang.lang.blocks import BlockType
        result = factory.block(BlockType.MCRY, [0], tg=1, mode="vchain",
                               theta=1.0)
        assert len(result) == 1
        assert result[0].symbol == "cry"

    def test_factory_mcrz_n1(self, factory):
        from lccfq_lang.lang.blocks import BlockType
        result = factory.block(BlockType.MCRZ, [0], tg=1, mode="vchain",
                               theta=1.0)
        assert len(result) == 1
        assert result[0].symbol == "crz"
