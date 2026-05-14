"""Equivalence helpers for mach-level optimization tests.

Provides a dense-vector simulator over the native gate set
{rx, ry, sqiswap}, plus assert_equivalent_native for tests.
Little-endian indexing: qubit 0 is LSB of state-vector index.
Measure / reset / nop / Control / Test are skipped (no semantic effect
on the simulated unitary trajectory).
"""
from __future__ import annotations
import numpy as np
from typing import List, Any
from lccfq_lang.mach.ir import Gate, Control, Test
from lccfq_lang.opt.op_view import OpView


def _u_rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _u_ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


# sqiswap matrix in the basis (|00>, |01>, |10>, |11>) where the FIRST
# index is the LOW qubit (little-endian convention shared with _sim.py).
# Reference: https://en.wikipedia.org/wiki/Quantum_logic_gate
_SQISWAP = np.array([
    [1.0,            0.0,            0.0, 0.0],
    [0.0, (1 + 1j) / 2,  (1 - 1j) / 2, 0.0],
    [0.0, (1 - 1j) / 2,  (1 + 1j) / 2, 0.0],
    [0.0,            0.0,            0.0, 1.0],
], dtype=complex)


def _apply_one_qubit(state: np.ndarray, q: int, n: int, u: np.ndarray) -> np.ndarray:
    new = np.zeros_like(state)
    for idx in range(1 << n):
        bit = (idx >> q) & 1
        partner = idx ^ (1 << q)
        if bit == 0:
            new[idx] += u[0, 0] * state[idx] + u[0, 1] * state[partner]
        else:
            new[idx] += u[1, 0] * state[partner] + u[1, 1] * state[idx]
    return new


def _apply_two_qubit_symmetric(
    state: np.ndarray,
    a: int,
    b: int,
    n: int,
    u4: np.ndarray,
) -> np.ndarray:
    """Apply a 4x4 unitary to qubits (a, b). Assumes the matrix is
    represented in the LOW-qubit-first basis: row index r is
    interpreted as r = (bit_b << 1) | bit_a."""
    if a == b:
        raise ValueError("_apply_two_qubit_symmetric: a == b")
    # For each computational basis index, compute the 2-bit local
    # index, look up the partners, accumulate.
    new = np.zeros_like(state)
    for idx in range(1 << n):
        bit_a = (idx >> a) & 1
        bit_b = (idx >> b) & 1
        local = (bit_b << 1) | bit_a   # in {0,1,2,3}
        for new_local in range(4):
            new_bit_a = new_local & 1
            new_bit_b = (new_local >> 1) & 1
            target_idx = idx
            if new_bit_a != bit_a:
                target_idx ^= (1 << a)
            if new_bit_b != bit_b:
                target_idx ^= (1 << b)
            new[target_idx] += u4[new_local, local] * state[idx]
    return new


def simulate_native(program: List[Any], n_qubits: int) -> np.ndarray:
    """Simulate a mach-level program; return the final statevector
    starting from |0...0>. Skips classical commands and reset/measure
    (their semantics are not modelled here).
    """
    state = np.zeros(1 << n_qubits, dtype=complex)
    state[0] = 1.0

    for op in program:
        if isinstance(op, (Control, Test)):
            continue
        if not isinstance(op, Gate):
            raise TypeError(f"simulate_native: unsupported op type {type(op).__name__}")
        sym = op.symbol
        if sym in ("nop", "measure", "reset"):
            continue
        if sym == "rx":
            q = OpView(op).qubits[0]
            state = _apply_one_qubit(state, q, n_qubits, _u_rx(op.params[0]))
        elif sym == "ry":
            q = OpView(op).qubits[0]
            state = _apply_one_qubit(state, q, n_qubits, _u_ry(op.params[0]))
        elif sym == "sqiswap":
            qs = OpView(op).qubits
            if len(qs) != 2:
                raise ValueError(f"simulate_native: sqiswap needs 2 qubits, got {qs}")
            a, b = qs[0], qs[1]
            state = _apply_two_qubit_symmetric(state, a, b, n_qubits, _SQISWAP)
        else:
            raise NotImplementedError(f"simulate_native: unsupported gate '{sym}'")
    return state


def assert_equivalent_native(
    p1: List[Any],
    p2: List[Any],
    n_qubits: int,
    tol: float = 1e-9,
) -> None:
    """Assert two mach programs produce the same statevector up to
    a global phase."""
    s1 = simulate_native(p1, n_qubits)
    s2 = simulate_native(p2, n_qubits)
    idx = int(np.argmax(np.abs(s1)))
    if abs(s1[idx]) < tol or abs(s2[idx]) < tol:
        fid = abs(np.vdot(s1, s2))
        assert abs(fid - 1.0) < tol, f"Programs not equivalent (fidelity={fid:.6f})"
        return
    phase = s2[idx] / s1[idx]
    phase = phase / abs(phase)
    s1_aligned = s1 * phase
    diff = np.max(np.abs(s1_aligned - s2))
    assert diff < tol, (
        f"Programs not equivalent up to global phase: max diff = {diff:.3e}"
    )
