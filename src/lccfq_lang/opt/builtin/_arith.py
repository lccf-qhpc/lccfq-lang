"""
Filename: _arith.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Numeric tolerances, angle arithmetic, and gate-class membership sets
    used by Phase 2 arch-level optimization passes.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
import math
from typing import FrozenSet

# Tolerance for "is angle zero" / "are angles equal" comparisons.
# Chosen to be tight relative to typical compile-time angle inputs (rationals
# of pi, products of small floats); loose enough not to flag genuine
# differences from optimization arithmetic accumulation.
ANGLE_TOL: float = 1e-10

# Canonical interval for reduced angles: (-pi, pi].
# Rationale: symmetric around zero, so identity rotations (theta ~ 0)
# do not get pushed near the wrap boundary by accumulation; numerically
# stable for common cases like sums of small rotations.
_PI = math.pi
_TWO_PI = 2.0 * math.pi


def MOD_2PI(theta: float) -> float:
    """Reduce *theta* into the canonical interval (-pi, pi].

    Edge cases:
      * theta exactly +pi maps to +pi (preserves the canonical right endpoint).
      * theta exactly -pi maps to +pi (since -pi == +pi modulo 2*pi, choose
        the closed end of the interval for determinism).
    """
    # math.remainder returns a value in [-pi, pi] (closed both sides).
    # Map -pi -> +pi to enforce the half-open interval (-pi, pi].
    r = math.remainder(theta, _TWO_PI)
    if r == -_PI:
        return _PI
    return r


def is_zero_angle(theta: float, tol: float = ANGLE_TOL) -> bool:
    """True iff MOD_2PI(theta) is within *tol* of 0."""
    return abs(MOD_2PI(theta)) <= tol


def is_equal_angle(a: float, b: float, tol: float = ANGLE_TOL) -> bool:
    """True iff a and b are equal modulo 2*pi (within *tol*)."""
    return is_zero_angle(a - b, tol)


# Self-inverse gates: g * g == I (up to global phase).
# Excludes s/sdg, t/tdg (their inverse is a *different* symbol — see INVERSE_PAIRS).
SELF_INVERSE: FrozenSet[str] = frozenset({"x", "y", "z", "h", "cx", "cz", "swap"})

# Gates that cancel pairwise across distinct symbols.
# Each inner frozenset is an unordered pair {a, b} with a*b == I (up to global phase).
INVERSE_PAIRS: FrozenSet[FrozenSet[str]] = frozenset({
    frozenset({"s", "sdg"}),
    frozenset({"t", "tdg"}),
})

# Single-qubit rotations that compose additively in their parameter:
# R(a) * R(b) == R(a+b) for the same rotation symbol on the same qubit.
#
# Note on "p": the LCCF ISA's p(theta) is the diagonal phase gate
# diag(1, exp(i*theta)). Two adjacent p gates compose: p(a) p(b) = p(a+b).
# This is identical to rz up to a global phase of exp(-i*theta/2), which
# does not affect the *structure* of the program (no global-phase tracking
# in this IR); for merging purposes p is treated like an additive rotation.
MERGEABLE_ROTATIONS: FrozenSet[str] = frozenset({"rx", "ry", "rz", "p"})


def inverse_symbol(sym: str) -> str | None:
    """Return the symbol *t* such that t * sym == I for distinct-symbol pairs.

    Returns None for self-inverse gates and for gates that have no
    fixed symbolic inverse in this ISA.
    """
    for pair in INVERSE_PAIRS:
        if sym in pair:
            other = next(iter(pair - {sym}))
            return other
    return None
