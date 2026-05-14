"""
Filename: _native.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Native gate-set membership constants for the XYiSW (rx / ry /
    sqiswap) target. Derived from xyisqswap._table; the only symbols
    that ever appear at mach level after transpilation are:
        - rx, ry  : single-qubit parametric rotations
        - sqiswap : two-qubit entangling primitive
        - nop, measure, reset : non-unitary or no-op commands
License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
from typing import FrozenSet

# Single-qubit parametric rotations on the device. Both axes compose
# additively on the same qubit, so MergeAdjacent1Q handles both.
NATIVE_1Q_PARAM: FrozenSet[str] = frozenset({"rx", "ry"})

# Two-qubit native entangling gates. sqiswap is symmetric on its qubit
# pair (the matrix is invariant under exchange of the two qubits — see
# the matrix definition in tests/opt/_equiv_native.py).
NATIVE_2Q: FrozenSet[str] = frozenset({"sqiswap"})

# Mach-level measurement op symbol. Confirmed by xyisqswap._table:
#     "measure": [("measure", [], ".")]
NATIVE_MEASURE: FrozenSet[str] = frozenset({"measure"})

# Reset is a non-unitary state-preparation primitive. Listed here for
# completeness; mach passes treat it like measure (an absorbing barrier
# for rotations on its qubit).
NATIVE_RESET: FrozenSet[str] = frozenset({"reset"})

# Convenience: the closed set of all symbols a mach-level program may
# carry. Useful for validation in tests.
NATIVE_ALL_SYMBOLS: FrozenSet[str] = (
    NATIVE_1Q_PARAM | NATIVE_2Q | NATIVE_MEASURE | NATIVE_RESET | frozenset({"nop"})
)
