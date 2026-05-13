"""
Filename: lower_universal.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Four fine-grained lowering passes that together replace QRegister.expand.
    These passes form the ``lower_expand`` PassGroup in the standard lowering
    pipeline.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
from typing import List
import numpy as np

from lccfq_lang.arch.instruction import Instruction
from lccfq_lang.arch.isa import ISA
from lccfq_lang.opt.pass_base import Pass, PassContext


LOWER_EXPAND_PASS_NAMES = ("lower_u2", "lower_u3", "lower_cu", "fanout_measure")


class LowerU2(Pass):
    """Decomposes u2(phi, lambda) into [rz(phi), ry(pi/2), rz(lambda)]."""

    name = "lower_u2"
    applies_to = "arch"

    def __init__(self, isa: ISA) -> None:
        self._isa = isa

    def run(self, program: List[Instruction], ctx: PassContext) -> List[Instruction]:
        out: List[Instruction] = []
        for instr in program:
            if instr.symbol == "u2":
                phi = instr.params[0]
                lbmd = instr.params[1]
                tg = instr.target_qubits[0]
                out.append(self._isa.rz(tg=tg, params=[phi]))
                out.append(self._isa.ry(tg=tg, params=[np.pi / 2]))
                out.append(self._isa.rz(tg=tg, params=[lbmd]))
            else:
                out.append(instr)
        return out


class LowerU3(Pass):
    """Decomposes u3(phi, theta, lambda) into [rz(phi), ry(theta), rz(lambda)]."""

    name = "lower_u3"
    applies_to = "arch"

    def __init__(self, isa: ISA) -> None:
        self._isa = isa

    def run(self, program: List[Instruction], ctx: PassContext) -> List[Instruction]:
        out: List[Instruction] = []
        for instr in program:
            if instr.symbol == "u3":
                phi = instr.params[0]
                theta = instr.params[1]
                lbmd = instr.params[2]
                tg = instr.target_qubits[0]
                out.append(self._isa.rz(tg=tg, params=[phi]))
                out.append(self._isa.ry(tg=tg, params=[theta]))
                out.append(self._isa.rz(tg=tg, params=[lbmd]))
            else:
                out.append(instr)
        return out


class LowerCU(Pass):
    """Decomposes cu(phi, theta, lambda) into a 7-instruction rz/ry/cx ladder.

    Mirrors register.expand:111-119 exactly.
    """

    name = "lower_cu"
    applies_to = "arch"

    def __init__(self, isa: ISA) -> None:
        self._isa = isa

    def run(self, program: List[Instruction], ctx: PassContext) -> List[Instruction]:
        out: List[Instruction] = []
        for instr in program:
            if instr.symbol == "cu":
                phi = instr.params[0]
                theta = instr.params[1]
                lbmd = instr.params[2]
                ct = instr.control_qubits[0]
                tg = instr.target_qubits[0]
                out.append(self._isa.rz(tg=tg, params=[lbmd]))
                out.append(self._isa.ry(tg=tg, params=[theta / 2]))
                out.append(self._isa.cx(ct=ct, tg=tg))
                out.append(self._isa.ry(tg=tg, params=[-theta / 2]))
                out.append(self._isa.rz(tg=tg, params=[-(phi + lbmd)]))
                out.append(self._isa.cx(ct=ct, tg=tg))
                out.append(self._isa.rz(tg=tg, params=[phi]))
            else:
                out.append(instr)
        return out


class FanoutMeasure(Pass):
    """Splits a multi-target measure into one single-target measure per qubit.

    A measure with len(target_qubits) <= 1 is forwarded unchanged.
    """

    name = "fanout_measure"
    applies_to = "arch"

    def __init__(self, isa: ISA) -> None:
        self._isa = isa

    def run(self, program: List[Instruction], ctx: PassContext) -> List[Instruction]:
        out: List[Instruction] = []
        for instr in program:
            if instr.symbol == "measure" and len(instr.target_qubits) > 1:
                for q in instr.target_qubits:
                    out.append(self._isa.measure(tgs=[q]))
            else:
                out.append(instr)
        return out
