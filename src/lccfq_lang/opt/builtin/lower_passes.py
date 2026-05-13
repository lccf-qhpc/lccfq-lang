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


LOWERING_STAGES: tuple[str, ...] = (
    "mapped",
    "swapped",
    "expanded",
    "arch_optimized",
    "transpiled",
)


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


def build_lowering_groups(
    qreg: QRegister,
    qpu,
    opt_level: int = 0,
    opt_passes: list[str] | None = None,
) -> list[PassGroup]:
    """Construct the lowering PassGroups for a given register and QPU.

    Phase 2 additions: when opt_level > 0 (or opt_passes is non-empty),
    insert an "arch_opt" PassGroup between "lower_expand" and "lower_transpile".

    :param qreg: virtual-to-physical mapping holder
    :param qpu: backend (provides isa, transpiler)
    :param opt_level: 0..3; ignored when opt_passes is not None
    :param opt_passes: explicit list of arch pass names; overrides opt_level
    """
    from .level_select import (
        passes_for_level,
        max_iters_for_level,
        resolve_opt_passes,
    )
    from .templates_arch import get_registered_templates

    groups: list[PassGroup] = [
        PassGroup("lower_map",  "linear", [MappedPass(qreg)]),
        PassGroup("lower_swap", "linear", [SwappedPass(qreg, qpu.isa)]),
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
    ]

    # Resolve arch_opt passes.
    if opt_passes is not None:
        if not isinstance(opt_passes, list):
            raise TypeError("build_lowering_groups: opt_passes must be a list[str] or None")
        arch_passes = resolve_opt_passes(opt_passes, qpu.isa)
        # Explicit-mode: do NOT auto-append registered user templates;
        # the user has stated their pass list exactly.
        max_iters = 5
    else:
        arch_passes = passes_for_level(opt_level, qpu.isa)
        # Implicit-mode: append user-registered templates when level >= 1.
        if opt_level >= 1:
            arch_passes = arch_passes + get_registered_templates()
        max_iters = max_iters_for_level(opt_level)

    if arch_passes:
        groups.append(
            PassGroup(
                "arch_opt",
                "fixpoint",
                arch_passes,
                max_iters=max_iters,
            )
        )

    groups.append(
        PassGroup("lower_transpile", "linear", [TranspiledPass(qpu.transpiler)])
    )
    return groups


def slice_groups_for(last_pass: str, groups: list[PassGroup]) -> list[PassGroup]:
    """Return the subset of *groups* needed to reach *last_pass*.

    :param last_pass: target pass name
    :param groups: full list of lowering PassGroups (in pipeline order)
    :return: groups[: idx + 1] where idx is the index of last_pass
    :raises UnknownCompilerPass: if last_pass is not "parsed", "executed",
        or one of LOWERING_STAGES
    """
    if last_pass == "parsed":
        return []
    if last_pass == "executed":
        return groups
    if last_pass not in LOWERING_STAGES:
        raise UnknownCompilerPass(last_pass)

    # Stage -> group name. arch_optimized is conditional.
    STAGE_TO_GROUP = {
        "mapped":         "lower_map",
        "swapped":        "lower_swap",
        "expanded":       "lower_expand",
        "arch_optimized": "arch_opt",
        "transpiled":     "lower_transpile",
    }
    target_group = STAGE_TO_GROUP[last_pass]

    # If the requested group is missing (arch_opt was omitted), fall back
    # to the immediately preceding lowering stage that *is* present.
    group_names = [g.name for g in groups]
    if target_group not in group_names:
        # Only arch_opt can be conditionally absent. Treat "arch_optimized"
        # as equivalent to "expanded" in this case.
        if last_pass == "arch_optimized":
            target_group = "lower_expand"
        else:
            # Should not occur: every other stage's group is unconditional.
            raise UnknownCompilerPass(last_pass)

    idx = group_names.index(target_group)
    return groups[: idx + 1]
