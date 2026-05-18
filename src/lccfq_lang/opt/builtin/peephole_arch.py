"""
Filename: peephole_arch.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Peephole optimization passes for the arch-level IR. All five passes
    operate on List[Instruction] and are pure (never mutate inputs).

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
from typing import List
from lccfq_lang.arch.instruction import Instruction
from lccfq_lang.arch.isa import ISA
from lccfq_lang.opt.pass_base import Pass, PassContext
from lccfq_lang.opt.op_view import OpView
from ._arith import (
    SELF_INVERSE,
    INVERSE_PAIRS,
    MERGEABLE_ROTATIONS,
    MOD_2PI,
    is_zero_angle,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _same_qubit_set(a: Instruction, b: Instruction) -> bool:
    """True iff two OpViews touch the same set of qubits (regardless of role)."""
    return set(OpView(a).qubits) == set(OpView(b).qubits)


def _same_role(a: Instruction, b: Instruction) -> bool:
    """True iff a and b have the same controls and targets, in the same role.

    Used for self-inverse cancellation: cx(0,1)·cx(0,1) cancels but
    cx(0,1)·cx(1,0) does NOT.
    """
    va, vb = OpView(a), OpView(b)
    return va.controls == vb.controls and va.targets == vb.targets


def _is_swap_match(a: Instruction, b: Instruction) -> bool:
    """SWAP is symmetric on its qubit set; cancellation should match
    swap(a,b) and swap(b,a). Compares qubit *sets*."""
    va, vb = OpView(a), OpView(b)
    return va.symbol == "swap" == vb.symbol and set(va.qubits) == set(vb.qubits)


# ---------------------------------------------------------------------------
# RemoveIdentity
# ---------------------------------------------------------------------------

class RemoveIdentity(Pass):
    """Drops nop and zero-angle single-qubit rotations."""

    name = "remove_identity"
    applies_to = "arch"

    def __init__(self, isa: ISA) -> None:
        # isa accepted for constructor symmetry with other arch passes;
        # not used inside run().
        self._isa = isa

    def run(self, program: List[Instruction], ctx: PassContext):
        out: List[Instruction] = []
        changed = False
        for instr in program:
            if instr.symbol == "nop":
                changed = True
                continue
            if (
                instr.symbol in MERGEABLE_ROTATIONS
                and instr.params is not None
                and len(instr.params) == 1
                and is_zero_angle(instr.params[0])
            ):
                changed = True
                continue
            out.append(instr)
        return out, changed


# ---------------------------------------------------------------------------
# CancelInverses
# ---------------------------------------------------------------------------

class CancelInverses(Pass):
    """Cancels adjacent self-inverse pairs and adjacent inverse-symbol pairs
    on the same qubits in the same role."""

    name = "cancel_inverses"
    applies_to = "arch"

    def __init__(self, isa: ISA) -> None:
        self._isa = isa

    def run(self, program: List[Instruction], ctx: PassContext):
        # `survivors` is a list of (idx, instr) where idx is the position
        # in the *output* list. last_op[q] holds the index in `survivors`
        # of the most recent surviving op touching qubit q, or -1.
        survivors: List[Instruction] = []
        last_op: dict[int, int] = {}
        changed = False

        for instr in program:
            qs = OpView(instr).qubits
            if not qs:
                # Op without qubit footprint (e.g. ftol). Cannot cancel, append.
                survivors.append(instr)
                continue

            # Find the previous op (must be the SAME op on every qubit in qs).
            prev_idx = last_op.get(qs[0], -1)
            if prev_idx >= 0 and all(last_op.get(q, -2) == prev_idx for q in qs):
                prev = survivors[prev_idx]
                if prev is not None and self._cancels(prev, instr):
                    # Remove prev. Mark its slot as a hole.
                    survivors[prev_idx] = None  # type: ignore[assignment]
                    changed = True
                    # Update last_op: scan back from prev_idx for each qubit in qs.
                    for q in OpView(prev).qubits:
                        new_last = self._previous_op_on(q, survivors, prev_idx)
                        if new_last < 0:
                            last_op.pop(q, None)
                        else:
                            last_op[q] = new_last
                    continue  # skip appending instr too

            # No cancellation: register instr as the new last_op for all its qubits.
            survivors.append(instr)
            new_idx = len(survivors) - 1
            for q in qs:
                last_op[q] = new_idx

        return [s for s in survivors if s is not None], changed

    @staticmethod
    def _cancels(a: Instruction, b: Instruction) -> bool:
        va, vb = OpView(a), OpView(b)
        # Self-inverse case: same symbol, same role (controls/targets).
        if va.symbol == vb.symbol and va.symbol in SELF_INVERSE:
            if va.symbol == "swap":
                return _is_swap_match(a, b)
            return _same_role(a, b)
        # Inverse-pair case: symbols form an INVERSE_PAIR, same controls/targets.
        if frozenset({va.symbol, vb.symbol}) in INVERSE_PAIRS:
            return _same_role(a, b)
        return False

    @staticmethod
    def _previous_op_on(q: int, survivors: list, before: int) -> int:
        """Walk backwards from before-1 to 0, return index of first op
        that touches q (skipping holes), or -1."""
        for i in range(before - 1, -1, -1):
            op = survivors[i]
            if op is None:
                continue
            if q in OpView(op).qubits:
                return i
        return -1


# ---------------------------------------------------------------------------
# MergeRotations
# ---------------------------------------------------------------------------

class MergeRotations(Pass):
    """Merges adjacent same-axis rotations on the same qubit:
    R(a) R(b) -> R(a+b). Drops the result entirely if a+b ~ 0 (mod 2*pi)."""

    name = "merge_rotations"
    applies_to = "arch"

    def __init__(self, isa: ISA) -> None:
        self._isa = isa

    def run(self, program: List[Instruction], ctx: PassContext):
        survivors: List[Instruction] = []
        last_op: dict[int, int] = {}
        changed = False

        for instr in program:
            qs = OpView(instr).qubits
            if not qs:
                survivors.append(instr)
                continue

            # Mergeable only if single qubit and a known additive rotation.
            if (
                len(qs) == 1
                and instr.symbol in MERGEABLE_ROTATIONS
                and instr.params is not None
                and len(instr.params) == 1
            ):
                q = qs[0]
                prev_idx = last_op.get(q, -1)
                if prev_idx >= 0:
                    prev = survivors[prev_idx]
                    if prev is not None:
                        pv = OpView(prev)
                        if (
                            prev.symbol == instr.symbol
                            and set(pv.qubits) == {q}
                            and prev.params is not None
                            and len(prev.params) == 1
                        ):
                            merged_angle = MOD_2PI(prev.params[0] + instr.params[0])
                            if is_zero_angle(merged_angle):
                                # Drop both.
                                survivors[prev_idx] = None  # type: ignore[assignment]
                                changed = True
                                new_last = self._previous_op_on(q, survivors, prev_idx)
                                if new_last < 0:
                                    last_op.pop(q, None)
                                else:
                                    last_op[q] = new_last
                                continue
                            # Replace prev with merged rotation built via ISA.
                            merged = self._build_rotation(instr.symbol, q, merged_angle)
                            survivors[prev_idx] = merged
                            last_op[q] = prev_idx
                            changed = True
                            continue

            survivors.append(instr)
            new_idx = len(survivors) - 1
            for q in qs:
                last_op[q] = new_idx

        return [s for s in survivors if s is not None], changed

    def _build_rotation(self, sym: str, q: int, angle: float) -> Instruction:
        method = getattr(self._isa, sym)
        return method(tg=q, params=[angle])

    @staticmethod
    def _previous_op_on(q: int, survivors: list, before: int) -> int:
        for i in range(before - 1, -1, -1):
            op = survivors[i]
            if op is None:
                continue
            if q in OpView(op).qubits:
                return i
        return -1


# ---------------------------------------------------------------------------
# FuseEulerZYZ
# ---------------------------------------------------------------------------

class FuseEulerZYZ(Pass):
    """Detects rz-ry-rz triplets on the same qubit and either eliminates them
    (if equivalent to identity) or normalises their angles into (-pi, pi]."""

    name = "fuse_euler_zyz"
    applies_to = "arch"

    def __init__(self, isa: ISA) -> None:
        self._isa = isa

    def run(self, program: List[Instruction], ctx: PassContext):
        # Build per-qubit ordered lists of indices, then sweep.
        # For simplicity, do a single forward pass tracking the last
        # *two* surviving ops per qubit.
        survivors: List[Instruction] = []
        # last1[q] = most recent surviving idx touching q
        # last2[q] = second-most recent surviving idx touching q,
        #           but only if no other op touched q between last2 and last1.
        last1: dict[int, int] = {}
        last2: dict[int, int] = {}
        changed = False

        for instr in program:
            qs = OpView(instr).qubits
            if (
                len(qs) == 1
                and instr.symbol == "rz"
                and instr.params is not None
                and len(instr.params) == 1
            ):
                q = qs[0]
                p_idx = last1.get(q, -1)
                pp_idx = last2.get(q, -1)
                if p_idx >= 0 and pp_idx >= 0:
                    p = survivors[p_idx]
                    pp = survivors[pp_idx]
                    if (
                        pp is not None and p is not None
                        and pp.symbol == "rz" and p.symbol == "ry"
                        and OpView(pp).qubits == (q,) and OpView(p).qubits == (q,)
                        and pp.params is not None and p.params is not None
                        and len(pp.params) == 1 and len(p.params) == 1
                    ):
                        alpha = pp.params[0]
                        beta = p.params[0]
                        gamma = instr.params[0]
                        if is_zero_angle(beta) and is_zero_angle(alpha + gamma):
                            # Drop pp, p, and instr entirely.
                            survivors[pp_idx] = None
                            survivors[p_idx] = None
                            changed = True
                            new_last = self._previous_op_on(q, survivors, pp_idx)
                            if new_last < 0:
                                last1.pop(q, None)
                                last2.pop(q, None)
                            else:
                                last1[q] = new_last
                                new_last2 = self._previous_op_on(q, survivors, new_last)
                                if new_last2 < 0:
                                    last2.pop(q, None)
                                else:
                                    last2[q] = new_last2
                            continue
                        # Normalise: replace pp, p; append normalised instr.
                        survivors[pp_idx] = self._isa.rz(tg=q, params=[MOD_2PI(alpha)])
                        survivors[p_idx] = self._isa.ry(tg=q, params=[MOD_2PI(beta)])
                        normalised = self._isa.rz(tg=q, params=[MOD_2PI(gamma)])
                        survivors.append(normalised)
                        new_idx = len(survivors) - 1
                        last2[q] = p_idx
                        last1[q] = new_idx
                        changed = True
                        continue

            survivors.append(instr)
            new_idx = len(survivors) - 1
            for q in qs:
                if q in last1:
                    last2[q] = last1[q]
                last1[q] = new_idx

        return [s for s in survivors if s is not None], changed

    @staticmethod
    def _previous_op_on(q: int, survivors: list, before: int) -> int:
        for i in range(before - 1, -1, -1):
            op = survivors[i]
            if op is None:
                continue
            if q in OpView(op).qubits:
                return i
        return -1


# ---------------------------------------------------------------------------
# CommuteThroughControl
# ---------------------------------------------------------------------------

class CommuteThroughControl(Pass):
    """Slides single-qubit rotations forward past commuting control gates
    to enable downstream merges and cancellations.

    DAG-based commutation deferred to Phase 2.5.
    """

    name = "commute_through_control"
    applies_to = "arch"

    _ROT_AXES = {"rx", "ry", "rz"}

    def __init__(self, isa: ISA) -> None:
        self._isa = isa

    def run(self, program: List[Instruction], ctx: PassContext):
        out: List[Instruction] = list(program)  # shallow copy; we mutate the list, not the elements
        changed = False
        i = 0
        while i + 1 < len(out):
            a, b = out[i], out[i + 1]
            if self._can_commute_forward(a, b):
                out[i], out[i + 1] = b, a
                changed = True
                # Advance past the swapped pair; do not re-examine the same
                # rotation against the next op (avoids oscillation).
                i += 2
            else:
                i += 1
        return out, changed

    @staticmethod
    def _can_commute_forward(a: Instruction, b: Instruction) -> bool:
        va, vb = OpView(a), OpView(b)
        if va.symbol not in CommuteThroughControl._ROT_AXES:
            return False
        if len(va.qubits) != 1:
            return False
        q = va.qubits[0]
        if vb.symbol == "cx":
            if len(vb.controls) != 1 or len(vb.targets) != 1:
                return False
            c, t = vb.controls[0], vb.targets[0]
            if va.symbol == "rz" and q == c:
                return True
            if va.symbol == "rx" and q == t:
                return True
            return False
        if vb.symbol == "cz":
            if len(vb.controls) != 1 or len(vb.targets) != 1:
                return False
            c, t = vb.controls[0], vb.targets[0]
            if va.symbol == "rz" and q in (c, t):
                return True
            return False
        return False
