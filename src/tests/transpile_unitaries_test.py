"""
Filename: transpile_unitaries_test.py
Author: Santiago Nunez-Corrales
Date: 2026-05-14
Version: 1.0
Description:
    Unitary-correctness tests for the XYiSW transpiler's two-qubit
    gate decompositions. For every controlled / SWAP gate, build the
    native sequence via Transpiler.transpile_gate, compose its 4x4
    unitary, and assert it matches the canonical reference up to
    global phase AND has matching Makhlin invariants.

License: Apache 2.0
"""
import pytest
import numpy as np
from math import pi as PI
from typing import List, Tuple

from lccfq_lang.arch.instruction import Instruction
from lccfq_lang.sys.factories.mach import TranspilerFactory


# ---------------------------------------------------------------------------
# Verification harness (§7 of spec)
# ---------------------------------------------------------------------------

_I = np.eye(2, dtype=complex)


def _rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)


# √iSWAP in (target=MSB, control=LSB) basis. From H = (X⊗X + Y⊗Y)/2 at t = π/4.
# Squares to iSWAP. Makhlin invariants (0, -1).
_SQISWAP = np.array([
    [1, 0,                  0,                  0],
    [0, 1.0 / np.sqrt(2),   1j  / np.sqrt(2),   0],
    [0, 1j  / np.sqrt(2),   1.0 / np.sqrt(2),   0],
    [0, 0,                  0,                  1],
], dtype=complex)


def _on_qubit(U2: np.ndarray, q: int, control_qubit: int, target_qubit: int) -> np.ndarray:
    """Embed a 1q matrix on the chosen qubit in the local |t,c> basis
    (target = MSB, control = LSB)."""
    if q == control_qubit:
        return np.kron(_I, U2)   # control = LSB
    elif q == target_qubit:
        return np.kron(U2, _I)   # target = MSB
    else:
        raise ValueError(f"qubit {q} is neither control nor target")


def compose_native_unitary(gates, control_qubit: int, target_qubit: int) -> np.ndarray:
    """Compose the local 4x4 unitary for a sequence of native gates.
    gates is a List[Gate] returned by Transpiler.transpile_gate.
    """
    U = np.eye(4, dtype=complex)
    for g in gates:
        sym = g.symbol
        if sym == "sqiswap":
            U = _SQISWAP @ U
            continue
        if sym in ("nop", "measure", "reset"):
            continue
        if sym == "rx":
            M = _rx(g.params[0])
        elif sym == "ry":
            M = _ry(g.params[0])
        elif sym == "rz":
            M = _rz(g.params[0])
        else:
            raise ValueError(f"compose_native_unitary: unsupported gate '{sym}'")
        q = g.target_qubits[0]
        U = _on_qubit(M, q, control_qubit, target_qubit) @ U
    return U


# --- Makhlin invariants (Makhlin 2002, Zhang et al 2003) ---
_B = (1 / np.sqrt(2)) * np.array([
    [1,   0,   0,  1j],
    [0,  1j,   1,   0],
    [0,  1j,  -1,   0],
    [1,   0,   0, -1j],
], dtype=complex)


def makhlin_invariants(U: np.ndarray) -> Tuple[complex, complex]:
    """Return (g1, g2) Makhlin local-equivalence invariants."""
    Ub = _B.conj().T @ U @ _B
    m = Ub.T @ Ub
    detU = np.linalg.det(U)
    g1 = np.trace(m) ** 2 / (16 * detU)
    g2 = (np.trace(m) ** 2 - np.trace(m @ m)) / (4 * detU)
    return complex(g1), complex(g2)


def assert_unitary_equiv_up_to_phase(U: np.ndarray, U_ref: np.ndarray,
                                      tol: float = 1e-9) -> None:
    """Assert U == U_ref up to global phase."""
    idx = np.unravel_index(np.argmax(np.abs(U_ref)), U_ref.shape)
    assert abs(U_ref[idx]) > tol, "reference matrix has no large entry"
    phase = U[idx] / U_ref[idx]
    assert abs(abs(phase) - 1.0) < tol, f"|U|/|U_ref| ratio not unit: {phase}"
    diff = np.max(np.abs(U - phase * U_ref))
    assert diff < tol, f"|U - phase*U_ref|_inf = {diff:.3e}"


def assert_makhlin_match(U: np.ndarray, expected: Tuple[float, float],
                          tol: float = 1e-9) -> None:
    g1, g2 = makhlin_invariants(U)
    g1e, g2e = expected
    assert abs(g1 - g1e) < tol, f"g1 mismatch: got {g1}, want {g1e}"
    assert abs(g2 - g2e) < tol, f"g2 mismatch: got {g2}, want {g2e}"


# ---------------------------------------------------------------------------
# Reference matrix builders (§7.1 of spec)
# ---------------------------------------------------------------------------

def cnot_ref():
    return np.array([[1, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0]], dtype=complex)


def cy_ref():
    M = np.zeros((4, 4), dtype=complex)
    M[0, 0] = 1
    M[2, 2] = 1
    M[3, 1] = 1j
    M[1, 3] = -1j
    return M


def cz_ref():
    return np.diag([1, 1, 1, -1]).astype(complex)


def ch_ref():
    s2 = 1 / np.sqrt(2)
    M = np.zeros((4, 4), dtype=complex)
    M[0, 0] = 1
    M[2, 2] = 1
    M[1, 1] = s2
    M[3, 1] = s2
    M[1, 3] = s2
    M[3, 3] = -s2
    return M


def swap_ref():
    M = np.zeros((4, 4), dtype=complex)
    M[0, 0] = 1
    M[1, 2] = 1
    M[2, 1] = 1
    M[3, 3] = 1
    return M


def cp_ref(theta):
    return np.diag([1, 1, 1, np.exp(1j * theta)]).astype(complex)


def crz_ref(theta):
    return np.diag([1, np.exp(-1j * theta / 2), 1, np.exp(1j * theta / 2)]).astype(complex)


def crx_ref(theta):
    M = np.eye(4, dtype=complex)
    rx = _rx(theta)
    M[1, 1] = rx[0, 0]
    M[1, 3] = rx[0, 1]
    M[3, 1] = rx[1, 0]
    M[3, 3] = rx[1, 1]
    return M


def cry_ref(theta):
    M = np.eye(4, dtype=complex)
    M[1, 1] = np.cos(theta / 2)
    M[1, 3] = -np.sin(theta / 2)
    M[3, 1] = np.sin(theta / 2)
    M[3, 3] = np.cos(theta / 2)
    return M


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

CTRL_Q = 0
TGT_Q = 1


def _transpile(symbol, params):
    instr = Instruction(
        symbol=symbol,
        target_qubits=[TGT_Q],
        control_qubits=[CTRL_Q],
        params=params,
        is_controlled=True,
    )
    transpiler = TranspilerFactory().get(mach="pfaff_v1")
    return transpiler.transpile_gate(instr)


# ---------------------------------------------------------------------------
# Non-parametric controlled / SWAP gates
# ---------------------------------------------------------------------------

NONPAR_CASES = [
    ("cx",   cnot_ref(), (0.0, 1.0)),
    ("cy",   cy_ref(),   (0.0, 1.0)),
    ("cz",   cz_ref(),   (0.0, 1.0)),
    ("ch",   ch_ref(),   (0.0, 1.0)),
    ("swap", swap_ref(), (-1.0, -3.0)),
]


@pytest.mark.parametrize("symbol,U_ref,makhlin", NONPAR_CASES)
def test_nonparametric_2q_unitary(symbol, U_ref, makhlin):
    gates = _transpile(symbol, params=None)
    U = compose_native_unitary(gates, CTRL_Q, TGT_Q)
    assert_unitary_equiv_up_to_phase(U, U_ref)
    assert_makhlin_match(U, makhlin)


# ---------------------------------------------------------------------------
# Parametric controlled gates
# ---------------------------------------------------------------------------

PARAM_THETAS = [0.0, PI / 8, PI / 4, PI / 2, PI, 3 * PI / 2, 2 * PI, -PI / 3, 2.7]


@pytest.mark.parametrize("theta", PARAM_THETAS)
def test_crz_unitary(theta):
    gates = _transpile("crz", params=[theta])
    U = compose_native_unitary(gates, CTRL_Q, TGT_Q)
    assert_unitary_equiv_up_to_phase(U, crz_ref(theta))


@pytest.mark.parametrize("theta", PARAM_THETAS)
def test_crx_unitary(theta):
    gates = _transpile("crx", params=[theta])
    U = compose_native_unitary(gates, CTRL_Q, TGT_Q)
    assert_unitary_equiv_up_to_phase(U, crx_ref(theta))


@pytest.mark.parametrize("theta", PARAM_THETAS)
def test_cry_unitary(theta):
    gates = _transpile("cry", params=[theta])
    U = compose_native_unitary(gates, CTRL_Q, TGT_Q)
    assert_unitary_equiv_up_to_phase(U, cry_ref(theta))


@pytest.mark.parametrize("theta", PARAM_THETAS)
def test_cp_unitary(theta):
    gates = _transpile("cp", params=[theta])
    U = compose_native_unitary(gates, CTRL_Q, TGT_Q)
    assert_unitary_equiv_up_to_phase(U, cp_ref(theta))


@pytest.mark.parametrize("theta", PARAM_THETAS)
def test_cphase_alias(theta):
    """cp and cphase MUST produce identical sequences (they are aliases)."""
    g_cp = _transpile("cp", params=[theta])
    g_cphase = _transpile("cphase", params=[theta])
    assert len(g_cp) == len(g_cphase)
    for a, b in zip(g_cp, g_cphase):
        assert a.symbol == b.symbol
        assert a.params == b.params
        assert a.target_qubits == b.target_qubits
        assert a.control_qubits == b.control_qubits


# ---------------------------------------------------------------------------
# Reverse-direction sanity check (control=1, target=0)
# ---------------------------------------------------------------------------

def test_cx_reversed_control_target():
    """Verify the decomposition is correct when control=1, target=0.

    compose_native_unitary always builds the local 4x4 in the (target=MSB,
    control=LSB) basis regardless of which physical qubit is control or
    target.  The CNOT reference in that basis is always cnot_ref() whether
    c=0,t=1 or c=1,t=0.
    """
    instr = Instruction(symbol="cx", target_qubits=[0], control_qubits=[1],
                        params=None, is_controlled=True)
    transpiler = TranspilerFactory().get(mach="pfaff_v1")
    gates = transpiler.transpile_gate(instr)
    U = compose_native_unitary(gates, control_qubit=1, target_qubit=0)
    assert_unitary_equiv_up_to_phase(U, cnot_ref())


# ---------------------------------------------------------------------------
# Callable params regression test (_synthesize extension)
# ---------------------------------------------------------------------------

def test_synthesize_callable_params():
    """Lambda p: [p[0]/2] must produce the derived angle, not the raw angle.

    CRZ(θ) expands to two CX sub-blocks each containing rx(π/2) on target,
    plus two Rz bands each containing one derived-angle rx on target.
    Filter to only the derived-angle gates (those whose |angle| != π/2).
    """
    theta = PI / 3
    gates = _transpile("crz", params=[theta])
    # Derived-angle rx gates on target are those with angle ≠ ±π/2.
    rx_on_t_derived = [
        g for g in gates
        if g.symbol == "rx"
        and TGT_Q in g.target_qubits
        and abs(abs(g.params[0]) - PI / 2) > 1e-9
    ]
    # Should have exactly two: one with -theta/2, one with +theta/2.
    assert len(rx_on_t_derived) == 2, (
        f"Expected 2 derived-angle rx on target, got {len(rx_on_t_derived)}: "
        f"{[g.params[0] for g in rx_on_t_derived]}"
    )
    vals = sorted(g.params[0] for g in rx_on_t_derived)
    assert abs(vals[0] - (-theta / 2)) < 1e-12, f"Expected -{theta/2}, got {vals[0]}"
    assert abs(vals[1] - (theta / 2)) < 1e-12, f"Expected {theta/2}, got {vals[1]}"
