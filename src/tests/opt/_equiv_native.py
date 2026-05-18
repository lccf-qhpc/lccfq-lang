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


# sqiswap matrix (actual √iSWAP, from H = (X⊗X + Y⊗Y)/2 evolved for t = π/4)
# in the basis (|00>, |01>, |10>, |11>) where the FIRST index is the LOW
# qubit (little-endian convention shared with _sim.py).
# Squares to iSWAP. Makhlin invariants (0, -1).
_SQISWAP = np.array([
    [1.0, 0.0,                  0.0,                  0.0],
    [0.0, 1.0 / np.sqrt(2),     1j  / np.sqrt(2),     0.0],
    [0.0, 1j  / np.sqrt(2),     1.0 / np.sqrt(2),     0.0],
    [0.0, 0.0,                  0.0,                  1.0],
], dtype=complex)


def _apply_one_qubit(state: np.ndarray, q: int, n: int, u: np.ndarray) -> np.ndarray:
    # Perf #8: vectorized via numpy fancy indexing.
    mask = 1 << q
    indices = np.arange(1 << n)
    zero_idx = indices[(indices & mask) == 0]
    one_idx = zero_idx | mask
    a = state[zero_idx]
    b = state[one_idx]
    new = state.copy()
    new[zero_idx] = u[0, 0] * a + u[0, 1] * b
    new[one_idx] = u[1, 0] * a + u[1, 1] * b
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
    interpreted as r = (bit_b << 1) | bit_a.

    Perf #8: vectorized. Partition the 2^n index space into 4 groups by
    (bit_b, bit_a). Each group's indices for a given local value form a
    contiguous slice in basis-state-space; apply the 4x4 matrix as four
    inner-product accumulations over the four (anchor, partner-quadruple)
    sets.
    """
    if a == b:
        raise ValueError("_apply_two_qubit_symmetric: a == b")
    mask_a = 1 << a
    mask_b = 1 << b
    indices = np.arange(1 << n)
    bit_a = (indices & mask_a) != 0
    bit_b = (indices & mask_b) != 0

    # Four index groups partitioned by local 2-bit value (bit_b<<1 | bit_a).
    # Each group has 2^(n-2) indices.
    idx_local_0 = indices[(~bit_a) & (~bit_b)]   # local = 00
    idx_local_1 = idx_local_0 | mask_a            # local = 01
    idx_local_2 = idx_local_0 | mask_b            # local = 10
    idx_local_3 = idx_local_0 | mask_a | mask_b   # local = 11
    locals_idx = [idx_local_0, idx_local_1, idx_local_2, idx_local_3]

    # Source amplitudes (one array per local input value).
    src = [state[i] for i in locals_idx]

    new = np.zeros_like(state)
    for new_local in range(4):
        # new[locals_idx[new_local]] = sum_{old_local} u4[new_local, old_local] * src[old_local]
        accum = np.zeros_like(src[0])
        for old_local in range(4):
            accum = accum + u4[new_local, old_local] * src[old_local]
        new[locals_idx[new_local]] = accum
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
