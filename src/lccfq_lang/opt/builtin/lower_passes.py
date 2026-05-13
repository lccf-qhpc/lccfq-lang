"""
Filename: lower_passes.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Built-in lowering passes for the lccfq-lang compilation pipeline.
    Provides MappedPass, SwappedPass, TranspiledPass,
    the build_lowering_groups factory, and the slice_groups_for slicer.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
from itertools import chain
from typing import List
from lccfq_lang.arch.instruction import Instruction
from lccfq_lang.arch.register import QRegister
from lccfq_lang.arch.isa import ISA
from lccfq_lang.arch.error import UnknownCompilerPass
from lccfq_lang.mach.transpilers import Transpiler
from lccfq_lang.opt.pass_base import Pass, PassContext
from lccfq_lang.opt.manager import PassGroup
from .lower_universal import LowerU2, LowerU3, LowerCU, FanoutMeasure


LOWERING_STAGES: tuple[str, ...] = ("mapped", "swapped", "expanded", "transpiled")


class MappedPass(Pass):
    """Maps virtual qubits to physical qubits."""

    name = "mapped"
    applies_to = "arch"

    def __init__(self, qreg: QRegister) -> None:
        self._qreg = qreg

    def run(self, program: List[Instruction], ctx: PassContext) -> List[Instruction]:
        return list(map(self._qreg.map, program))


class SwappedPass(Pass):
    """Inserts SWAP gates to satisfy connectivity constraints."""

    name = "swapped"
    applies_to = "arch"

    def __init__(self, qreg: QRegister, isa: ISA) -> None:
        self._qreg = qreg
        self._isa = isa

    def run(self, program: List[Instruction], ctx: PassContext) -> List[Instruction]:
        return list(chain.from_iterable(
            map(lambda instr: self._qreg.swaps(instr, self._isa), program)
        ))


class TranspiledPass(Pass):
    """Transpiles arch instructions to native machine gates.

    applies_to is "arch" because the input to this pass is List[Instruction].
    PassManager._run_linear type-checks the input (not the output), so labeling
    this "mach" would cause a type error. The output (List[Gate]) ends the
    pipeline; no subsequent group inspects it.
    """

    name = "transpiled"
    applies_to = "arch"

    def __init__(self, transpiler: Transpiler) -> None:
        self._transpiler = transpiler

    def run(self, program: List[Instruction], ctx: PassContext) -> list:
        return list(chain.from_iterable(
            map(self._transpiler.transpile_gate, program)
        ))


def build_lowering_groups(qreg: QRegister, qpu) -> list[PassGroup]:
    """Construct the four standard lowering PassGroups for a given register and QPU."""
    return [
        PassGroup("lower_map",       "linear", [MappedPass(qreg)]),
        PassGroup("lower_swap",      "linear", [SwappedPass(qreg, qpu.isa)]),
        PassGroup(
            "lower_expand",
            "linear",
            [
                LowerU2(qpu.isa),
                LowerU3(qpu.isa),
                LowerCU(qpu.isa),
                FanoutMeasure(qpu.isa),
            ],
        ),
        PassGroup("lower_transpile", "linear", [TranspiledPass(qpu.transpiler)]),
    ]


def slice_groups_for(last_pass: str, groups: list[PassGroup]) -> list[PassGroup]:
    """Return the subset of *groups* needed to reach *last_pass*.

    :param last_pass: target pass name
    :param groups: full list of lowering PassGroups (in pipeline order)
    :return: groups[: idx + 1] where idx is the index of last_pass
    :raises UnknownCompilerPass: if last_pass is not "parsed", "executed", or
        one of LOWERING_STAGES
    """
    if last_pass == "parsed":
        return []
    if last_pass == "executed":
        return groups
    if last_pass not in LOWERING_STAGES:
        raise UnknownCompilerPass(last_pass)
    idx = LOWERING_STAGES.index(last_pass)
    return groups[: idx + 1]
