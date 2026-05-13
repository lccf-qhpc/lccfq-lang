"""Equivalence helpers for arch-level optimization tests.

Wraps src/tests/_sim.simulate to run two programs and assert the resulting
state vectors are equal up to global phase.
"""
from __future__ import annotations
import numpy as np
from typing import List
from lccfq_lang.arch.instruction import Instruction
from tests._sim import simulate


def simulate_program(program: List[Instruction], n_qubits: int) -> np.ndarray:
    """Run the program on |0...0> and return the final statevector."""
    return simulate(program, n_qubits)


def assert_equivalent(
    p1: List[Instruction],
    p2: List[Instruction],
    n_qubits: int,
    tol: float = 1e-9,
) -> None:
    """Assert p1 and p2 produce the same statevector up to a global phase."""
    s1 = simulate_program(p1, n_qubits)
    s2 = simulate_program(p2, n_qubits)
    # Find a non-zero entry to extract relative phase.
    idx = int(np.argmax(np.abs(s1)))
    if abs(s1[idx]) < tol or abs(s2[idx]) < tol:
        # Fall back to direct fidelity measure.
        fid = abs(np.vdot(s1, s2))
        assert abs(fid - 1.0) < tol, (
            f"Programs not equivalent (fidelity={fid:.6f})"
        )
        return
    phase = s2[idx] / s1[idx]
    phase = phase / abs(phase)
    s1_aligned = s1 * phase
    diff = np.max(np.abs(s1_aligned - s2))
    assert diff < tol, (
        f"Programs not equivalent up to global phase: max diff = {diff:.3e}"
    )
