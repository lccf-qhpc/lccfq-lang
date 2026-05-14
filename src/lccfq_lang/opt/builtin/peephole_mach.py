"""
Filename: peephole_mach.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Mach-level peephole optimization passes for the lccfq-lang compilation
    pipeline. Operates on List[mach.ir.Gate | Control | Test] produced by
    XYiSW.transpile_gate.

    Three passes:
        RemoveIdentityMach  — drop nop and zero-angle rx/ry
        MergeAdjacent1Q     — merge adjacent same-symbol rotations on same qubit
        EulerXYRecompose    — collapse runs of >= 4 rotations into Ry-Rx-Ry form

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
import math
import numpy as np
from typing import Optional
from lccfq_lang.arch.isa import ISA
from lccfq_lang.mach.ir import Gate
from lccfq_lang.opt.pass_base import Pass, PassContext
from lccfq_lang.opt.op_view import OpView
from ._arith import MOD_2PI, is_zero_angle, ANGLE_TOL
from ._native import NATIVE_1Q_PARAM


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _previous_op_on(q: int, survivors: list, before: int) -> int:
    """Walk backwards from before-1 to 0, return index of first op that
    touches qubit q (skipping None holes), or -1 if none found."""
    for i in range(before - 1, -1, -1):
        op = survivors[i]
        if op is None:
            continue
        if q in OpView(op).qubits:
            return i
    return -1


# ---------------------------------------------------------------------------
# _xyx_decompose
# ---------------------------------------------------------------------------

def _xyx_decompose(U: np.ndarray) -> tuple[float, float, float]:
    """Return (a, b, c) such that U = e^{iphi} * Ry(c) Rx(b) Ry(a).

    Uses the standard SU(2) Euler decomposition with axes Y-X-Y.
    Returns angles in (-pi, pi] (the canonical interval used by
    MOD_2PI). Numerically stable for b near 0 and near pi.

    Algorithm:
      1. Strip global phase: M = U / sqrt(det U).
      2. b = 2 * acos(clip(|M[0,0] + M[1,1]| / 2, 0, 1))
      3. Use atan2-based formulas for a and c that are valid away
         from b ~ 0 and b ~ pi; fall back to fused-angle formulas at
         the singular points.
    """
    # Step 1: strip global phase to get M in SU(2)
    det = U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0]
    phase = np.sqrt(det)
    if abs(phase) < 1e-14:
        return 0.0, 0.0, 0.0
    M = U / phase

    re00, im00 = M[0, 0].real, M[0, 0].imag
    re10, im10 = M[1, 0].real, M[1, 0].imag

    cb = math.sqrt(re00 ** 2 + re10 ** 2)
    sb = math.sqrt(im00 ** 2 + im10 ** 2)
    b = 2.0 * math.atan2(sb, cb)   # in [0, pi]

    if sb < 1e-9:
        # b ~ 0: only p = (a+c)/2 is determined; set a=0, c=2p.
        p = math.atan2(re10, re00)   # cb*sin(p)/cb*cos(p)
        a = 0.0
        c = 2.0 * p
    elif cb < 1e-9:
        # b ~ pi: only r = (c-a)/2 is determined; set c=0, a=-2r.
        r = math.atan2(im00, -im10)  # sb*sin(r)/sb*cos(r), with Im(M10)=-sb*cos(r)
        c = 0.0
        a = -2.0 * r
    else:
        p = math.atan2(re10, re00)    # (a+c)/2
        r = math.atan2(im00, -im10)   # (c-a)/2  [since Im(M10)=-sb*cos(r)]
        a = p - r
        c = p + r

    return MOD_2PI(a), MOD_2PI(b), MOD_2PI(c)


# ---------------------------------------------------------------------------
# RemoveIdentityMach
# ---------------------------------------------------------------------------

class RemoveIdentityMach(Pass):
    """Drops nop and zero-angle native single-qubit rotations.

    Symbols handled: nop, rx, ry. Two-qubit gates and measurements are
    never identity in this IR (sqiswap is not parametric).
    """
    name = "remove_identity_mach"
    applies_to = "mach"

    def __init__(self, isa: ISA) -> None:
        # isa kept for constructor symmetry; not used.
        self._isa = isa

    def run(self, program, ctx):
        out = []
        for op in program:
            if isinstance(op, Gate):
                if op.symbol == "nop":
                    continue
                if (
                    op.symbol in NATIVE_1Q_PARAM
                    and op.params is not None
                    and len(op.params) == 1
                    and is_zero_angle(op.params[0])
                ):
                    continue
            out.append(op)
        return out


# ---------------------------------------------------------------------------
# MergeAdjacent1Q
# ---------------------------------------------------------------------------

class MergeAdjacent1Q(Pass):
    """Merges adjacent same-symbol rotations on the same qubit.

    rx(a) rx(b) -> rx(a+b); same for ry. Drops the result entirely
    when (a+b) ~ 0 (mod 2pi). Does NOT mix axes (rx and ry on the
    same qubit are NOT merged here; see EulerXYRecompose).
    """
    name = "merge_adjacent_1q"
    applies_to = "mach"

    def __init__(self, isa: ISA) -> None:
        self._isa = isa

    def run(self, program, ctx):
        survivors: list = []
        last_op: dict[int, int] = {}

        for op in program:
            qs = OpView(op).qubits
            if not qs:
                survivors.append(op)
                continue

            mergeable = (
                isinstance(op, Gate)
                and len(qs) == 1
                and op.symbol in NATIVE_1Q_PARAM
                and op.params is not None
                and len(op.params) == 1
            )
            if mergeable:
                q = qs[0]
                prev_idx = last_op.get(q, -1)
                if prev_idx >= 0:
                    prev = survivors[prev_idx]
                    if (
                        prev is not None
                        and isinstance(prev, Gate)
                        and prev.symbol == op.symbol
                        and set(OpView(prev).qubits) == {q}
                        and prev.params is not None
                        and len(prev.params) == 1
                    ):
                        merged_angle = MOD_2PI(prev.params[0] + op.params[0])
                        if is_zero_angle(merged_angle):
                            survivors[prev_idx] = None
                            new_last = _previous_op_on(q, survivors, prev_idx)
                            if new_last < 0:
                                last_op.pop(q, None)
                            else:
                                last_op[q] = new_last
                            continue
                        merged = Gate(
                            symbol=op.symbol,
                            target_qubits=list(prev.target_qubits),
                            control_qubits=prev.control_qubits,
                            params=[merged_angle],
                        )
                        survivors[prev_idx] = merged
                        last_op[q] = prev_idx
                        continue

            survivors.append(op)
            new_idx = len(survivors) - 1
            for q in qs:
                last_op[q] = new_idx

        return [s for s in survivors if s is not None]


# ---------------------------------------------------------------------------
# EulerXYRecompose
# ---------------------------------------------------------------------------

class EulerXYRecompose(Pass):
    """Collapses runs of >=4 single-qubit rotations on one qubit into
    canonical Ry(a) Rx(b) Ry(c). Conservative: only fires when the
    rewrite strictly reduces gate count."""

    name = "euler_xy_recompose"
    applies_to = "mach"

    def __init__(self, isa: ISA) -> None:
        self._isa = isa

    def run(self, program, ctx):
        out = list(program)
        n = len(out)
        i = 0
        result: list = []
        while i < n:
            op = out[i]
            run_q = self._lone_rotation_qubit(op)
            if run_q is None:
                result.append(op)
                i += 1
                continue
            # Greedily extend the run.
            j = i + 1
            while j < n and self._lone_rotation_qubit(out[j]) == run_q:
                j += 1
            run = out[i:j]
            if len(run) >= 4:
                rewritten = self._try_recompose(run, run_q)
                if rewritten is not None and len(rewritten) < len(run):
                    result.extend(rewritten)
                else:
                    result.extend(run)
            else:
                result.extend(run)
            i = j
        return result

    @staticmethod
    def _lone_rotation_qubit(op):
        if not isinstance(op, Gate):
            return None
        if op.symbol not in NATIVE_1Q_PARAM:
            return None
        qs = OpView(op).qubits
        if len(qs) != 1:
            return None
        return qs[0]

    def _try_recompose(self, run, q):
        U = np.eye(2, dtype=complex)
        for g in run:
            U = self._gate_matrix(g) @ U
        a, b, c = _xyx_decompose(U)
        rewritten: list = []
        # Decomposition gives U = Ry(c) Rx(b) Ry(a), where Ry(a) is
        # applied FIRST. Therefore program order is [Ry(a), Rx(b), Ry(c)].
        if not is_zero_angle(a):
            rewritten.append(Gate("ry", [q], None, [MOD_2PI(a)]))
        if not is_zero_angle(b):
            rewritten.append(Gate("rx", [q], None, [MOD_2PI(b)]))
        if not is_zero_angle(c):
            rewritten.append(Gate("ry", [q], None, [MOD_2PI(c)]))
        return rewritten

    @staticmethod
    def _gate_matrix(g):
        theta = g.params[0]
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        if g.symbol == "rx":
            return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        if g.symbol == "ry":
            return np.array([[c, -s], [s, c]], dtype=complex)
        raise ValueError(f"_gate_matrix: unsupported {g.symbol}")
