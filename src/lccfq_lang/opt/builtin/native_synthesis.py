"""
Filename: native_synthesis.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Native synthesis passes for the mach-level IR.

    RyRzRyToHardware — collapses the canonical 6-gate band emitted by
    transpiling two adjacent virtual rz instructions on the same qubit.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
import math
from lccfq_lang.arch.isa import ISA
from lccfq_lang.mach.ir import Gate
from lccfq_lang.opt.pass_base import Pass, PassContext
from lccfq_lang.opt.op_view import OpView
from ._arith import MOD_2PI, ANGLE_TOL, is_equal_angle


class RyRzRyToHardware(Pass):
    """Recognizes the canonical 6-gate band emitted by transpiling two
    adjacent virtual rz instructions on the same qubit, and collapses
    the inner ry(+pi/2) ry(-pi/2) annihilation into a 3-gate band
    representing rz(alpha+beta).

    Pattern (all six on the same qubit q, contiguous in program order):
        ry(-pi/2), rx(alpha), ry(+pi/2),
        ry(-pi/2), rx(beta),  ry(+pi/2)
    Rewrite:
        ry(-pi/2), rx(alpha+beta), ry(+pi/2)
    """
    name = "ry_rz_ry_to_hardware"
    applies_to = "mach"

    _HALF_PI_TOL = ANGLE_TOL

    def __init__(self, isa: ISA) -> None:
        self._isa = isa

    def run(self, program, ctx):
        out: list = []
        changed = False
        i = 0
        n = len(program)
        while i + 5 < n:
            window = program[i:i + 6]
            qa = self._matches(window)
            if qa is not None:
                q, alpha_plus_beta = qa
                out.append(Gate("ry", [q], None, [-math.pi / 2]))
                out.append(Gate("rx", [q], None, [MOD_2PI(alpha_plus_beta)]))
                out.append(Gate("ry", [q], None, [+math.pi / 2]))
                changed = True
                i += 6
                continue
            out.append(program[i])
            i += 1
        # Trailing tail (last <6 ops).
        out.extend(program[i:])
        return out, changed

    @staticmethod
    def _matches(window):
        # window is exactly 6 ops. Returns (q, alpha+beta) or None.
        if not all(isinstance(op, Gate) for op in window):
            return None
        # Symbol pattern check
        symbols = [op.symbol for op in window]
        if symbols != ["ry", "rx", "ry", "ry", "rx", "ry"]:
            return None
        # Same single qubit on every op
        qubits = [OpView(op).qubits for op in window]
        if any(len(qs) != 1 for qs in qubits):
            return None
        q = qubits[0][0]
        if any(qs[0] != q for qs in qubits):
            return None
        # Angle pattern check
        params = [op.params for op in window]
        if any(p is None or len(p) != 1 for p in params):
            return None
        # ry(-pi/2), rx(alpha), ry(+pi/2), ry(-pi/2), rx(beta), ry(+pi/2)
        if not is_equal_angle(params[0][0], -math.pi / 2):
            return None
        if not is_equal_angle(params[2][0], +math.pi / 2):
            return None
        if not is_equal_angle(params[3][0], -math.pi / 2):
            return None
        if not is_equal_angle(params[5][0], +math.pi / 2):
            return None
        alpha = params[1][0]
        beta = params[4][0]
        return q, alpha + beta
