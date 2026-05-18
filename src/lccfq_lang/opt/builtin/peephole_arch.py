"""
Filename: peephole_arch.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Peephole optimization passes for the arch-level IR. All five passes
    operate on List[Instruction] and are pure (never mutate inputs).

    Perf #7 (2026-05-18): CancelInverses, MergeRotations, and FuseEulerZYZ
    now use per-qubit doubly-linked-list survivor nodes (_SurvivorNode from
    _survivor_chain.py) for O(1) "previous op on qubit q" lookup, replacing
    the former O(n) backward-scan _previous_op_on helpers.

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
from ._survivor_chain import _SurvivorNode, append_node, cancel_node


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
    on the same qubits in the same role.

    Perf #7: uses per-qubit doubly-linked-list (_SurvivorNode) for O(1)
    previous-op-on-qubit lookup.  last_node_on_q[q] always points to the
    most-recent surviving node that touches qubit q.
    """

    name = "cancel_inverses"
    applies_to = "arch"

    def __init__(self, isa: ISA) -> None:
        self._isa = isa

    def run(self, program: List[Instruction], ctx: PassContext):
        survivors: list[_SurvivorNode] = []
        last_node_on_q: dict[int, _SurvivorNode] = {}
        changed = False

        for instr in program:
            qs = OpView(instr).qubits
            if not qs:
                # Op without qubit footprint (e.g. ftol). Cannot cancel, append.
                # Does NOT enter any per-qubit chain (append_node skips empty qs).
                append_node(survivors, last_node_on_q, instr)
                continue

            # Find the previous op: must be the SAME node on every qubit in qs.
            prev_node = last_node_on_q.get(qs[0])
            if (
                prev_node is not None
                and all(last_node_on_q.get(q) is prev_node for q in qs)
                and self._cancels(prev_node.op, instr)
            ):
                # Cancel: mark dead and splice out of all per-qubit chains.
                cancel_node(prev_node, last_node_on_q)
                changed = True
                # Do NOT append instr — both ops vanish.
                continue

            # No cancellation: append instr as a new surviving node.
            append_node(survivors, last_node_on_q, instr)

        return [node.op for node in survivors if node.alive], changed

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


# ---------------------------------------------------------------------------
# MergeRotations
# ---------------------------------------------------------------------------

class MergeRotations(Pass):
    """Merges adjacent same-axis rotations on the same qubit:
    R(a) R(b) -> R(a+b). Drops the result entirely if a+b ~ 0 (mod 2*pi).

    Perf #7: uses per-qubit doubly-linked-list for O(1) previous-op lookup.
    Replace-in-place (node.op reassignment) preserves the node's qubit
    footprint so chains remain valid.
    """

    name = "merge_rotations"
    applies_to = "arch"

    def __init__(self, isa: ISA) -> None:
        self._isa = isa

    def run(self, program: List[Instruction], ctx: PassContext):
        survivors: list[_SurvivorNode] = []
        last_node_on_q: dict[int, _SurvivorNode] = {}
        changed = False

        for instr in program:
            qs = OpView(instr).qubits
            if not qs:
                append_node(survivors, last_node_on_q, instr)
                continue

            # Mergeable only if single qubit and a known additive rotation.
            if (
                len(qs) == 1
                and instr.symbol in MERGEABLE_ROTATIONS
                and instr.params is not None
                and len(instr.params) == 1
            ):
                q = qs[0]
                prev_node = last_node_on_q.get(q)

                if prev_node is not None:
                    prev = prev_node.op
                    pv = OpView(prev)
                    if (
                        prev.symbol == instr.symbol
                        and set(pv.qubits) == {q}
                        and prev.params is not None
                        and len(prev.params) == 1
                    ):
                        merged_angle = MOD_2PI(prev.params[0] + instr.params[0])
                        if is_zero_angle(merged_angle):
                            # Drop both: cancel the prev node, skip instr.
                            cancel_node(prev_node, last_node_on_q)
                            changed = True
                            continue
                        # Replace prev with merged rotation (replace-in-place).
                        # Invariant: qubit footprint unchanged — same single qubit q.
                        merged = self._build_rotation(instr.symbol, q, merged_angle)
                        assert OpView(merged).qubits == OpView(prev_node.op).qubits, (
                            "replace-in-place changed qubit footprint"
                        )
                        prev_node.op = merged
                        # last_node_on_q[q] still points to prev_node — correct.
                        changed = True
                        continue

            append_node(survivors, last_node_on_q, instr)

        return [node.op for node in survivors if node.alive], changed

    def _build_rotation(self, sym: str, q: int, angle: float) -> Instruction:
        method = getattr(self._isa, sym)
        return method(tg=q, params=[angle])


# ---------------------------------------------------------------------------
# FuseEulerZYZ
# ---------------------------------------------------------------------------

class FuseEulerZYZ(Pass):
    """Detects rz-ry-rz triplets on the same qubit and either eliminates them
    (if equivalent to identity) or normalises their angles into (-pi, pi].

    Perf #7: last1[q] / last2[q] are now derived on demand from
    last_node_on_q[q] and its prev_per_qubit[q] pointer, each an O(1) read.
    The two formerly chained _previous_op_on calls (O(n) each) are gone.
    """

    name = "fuse_euler_zyz"
    applies_to = "arch"

    def __init__(self, isa: ISA) -> None:
        self._isa = isa

    def run(self, program: List[Instruction], ctx: PassContext):
        survivors: list[_SurvivorNode] = []
        last_node_on_q: dict[int, _SurvivorNode] = {}
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
                # O(1) reads: most-recent and second-most-recent nodes on q.
                p_node = last_node_on_q.get(q)
                pp_node = p_node.prev_per_qubit.get(q) if p_node is not None else None

                if p_node is not None and pp_node is not None:
                    p = p_node.op
                    pp = pp_node.op

                    if (
                        pp.symbol == "rz" and p.symbol == "ry"
                        and OpView(pp).qubits == (q,) and OpView(p).qubits == (q,)
                        and pp.params is not None and p.params is not None
                        and len(pp.params) == 1 and len(p.params) == 1
                    ):
                        alpha = pp.params[0]
                        beta = p.params[0]
                        gamma = instr.params[0]
                        if is_zero_angle(beta) and is_zero_angle(alpha + gamma):
                            # Drop pp_node and p_node; skip instr entirely.
                            # Cancel pp_node first (deeper in the chain).
                            cancel_node(pp_node, last_node_on_q)
                            # After removing pp_node, p_node is now head on q
                            # (last_node_on_q[q] == p_node after the first cancel).
                            cancel_node(p_node, last_node_on_q)
                            changed = True
                            continue
                        # Normalise: replace pp and p in-place; append normalised instr.
                        norm_pp = self._isa.rz(tg=q, params=[MOD_2PI(alpha)])
                        norm_p = self._isa.ry(tg=q, params=[MOD_2PI(beta)])
                        assert OpView(norm_pp).qubits == OpView(pp_node.op).qubits, (
                            "replace-in-place changed qubit footprint (pp)"
                        )
                        assert OpView(norm_p).qubits == OpView(p_node.op).qubits, (
                            "replace-in-place changed qubit footprint (p)"
                        )
                        pp_node.op = norm_pp
                        p_node.op = norm_p
                        # Append normalised gamma as a new node.
                        normalised = self._isa.rz(tg=q, params=[MOD_2PI(gamma)])
                        append_node(survivors, last_node_on_q, normalised)
                        changed = True
                        continue

            # Default: append as new surviving node.
            append_node(survivors, last_node_on_q, instr)

        return [node.op for node in survivors if node.alive], changed


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
