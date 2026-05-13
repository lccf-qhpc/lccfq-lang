"""
Filename: templates_arch.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Named template-rewrite passes for the arch_opt group, plus the
    user-facing TEMPLATE_REGISTRY for external rewrites.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
from typing import List
from lccfq_lang.arch.instruction import Instruction
from lccfq_lang.arch.isa import ISA
from lccfq_lang.opt.pass_base import Pass, PassContext
from lccfq_lang.opt.op_view import OpView


# ---------------------------------------------------------------------------
# HCXHRule
# ---------------------------------------------------------------------------

class HCXHRule(Pass):
    """Collapses H(t) CX(c,t) H(t) into CZ(c,t)."""

    name = "hcxh_to_cz"
    applies_to = "arch"

    def __init__(self, isa: ISA) -> None:
        self._isa = isa

    def run(self, program: List[Instruction], ctx: PassContext) -> List[Instruction]:
        # Walk with index; on a match, append the CZ and skip the three matched ops.
        out: List[Instruction] = []
        i = 0
        n = len(program)
        while i < n:
            if i + 2 < n:
                a, b, c = program[i], program[i + 1], program[i + 2]
                t = self._match_hcxh(a, b, c)
                if t is not None:
                    ctrl = b.control_qubits[0]
                    out.append(self._isa.cz(ct=ctrl, tg=t))
                    i += 3
                    continue
            out.append(program[i])
            i += 1
        return out

    @staticmethod
    def _match_hcxh(a: Instruction, b: Instruction, c: Instruction) -> int | None:
        va, vb, vc = OpView(a), OpView(b), OpView(c)
        if va.symbol != "h" or vc.symbol != "h":
            return None
        if vb.symbol != "cx":
            return None
        if len(va.targets) != 1 or len(vc.targets) != 1:
            return None
        t1, t3 = va.targets[0], vc.targets[0]
        if t1 != t3:
            return None
        if len(vb.targets) != 1 or len(vb.controls) != 1:
            return None
        if vb.targets[0] != t1:
            return None
        return t1


# ---------------------------------------------------------------------------
# SwapElision
# ---------------------------------------------------------------------------

class SwapElision(Pass):
    """Drops adjacent SWAP pairs on the same qubit set."""

    name = "swap_elision"
    applies_to = "arch"

    def __init__(self, isa: ISA) -> None:
        self._isa = isa

    def run(self, program: List[Instruction], ctx: PassContext) -> List[Instruction]:
        out: List[Instruction] = []
        i = 0
        n = len(program)
        while i < n:
            if i + 1 < n:
                a, b = program[i], program[i + 1]
                if (
                    a.symbol == "swap"
                    and b.symbol == "swap"
                    and set(OpView(a).qubits) == set(OpView(b).qubits)
                    and len(OpView(a).qubits) == 2
                ):
                    i += 2
                    continue
            out.append(program[i])
            i += 1
        return out


# ---------------------------------------------------------------------------
# TEMPLATE_REGISTRY and helpers
# ---------------------------------------------------------------------------

# Module-level registry. Keys are template names (Pass.name); values are
# Pass instances ready to be inserted into the arch_opt group.
TEMPLATE_REGISTRY: dict[str, Pass] = {}


def register_template(name: str, pass_obj: Pass) -> None:
    """Register a user-supplied arch-level template-rewrite pass.

    :param name: unique identifier; must match pass_obj.name.
    :param pass_obj: a Pass instance with applies_to == "arch".
    :raises TypeError: if name is not a str or pass_obj is not a Pass.
    :raises ValueError: if pass_obj.applies_to != "arch", if name does not
        match pass_obj.name, or if name is already registered.
    """
    if not isinstance(name, str) or not name:
        raise TypeError("register_template: name must be a non-empty string")
    if not isinstance(pass_obj, Pass):
        raise TypeError(
            f"register_template: pass_obj must be a Pass, got {type(pass_obj).__name__}"
        )
    if pass_obj.applies_to != "arch":
        raise ValueError(
            f"register_template: pass_obj.applies_to must be 'arch', "
            f"got {pass_obj.applies_to!r}"
        )
    if pass_obj.name != name:
        raise ValueError(
            f"register_template: name {name!r} does not match pass_obj.name {pass_obj.name!r}"
        )
    if name in TEMPLATE_REGISTRY:
        raise ValueError(f"register_template: name {name!r} is already registered")
    TEMPLATE_REGISTRY[name] = pass_obj


def unregister_template(name: str) -> None:
    """Remove a previously registered template (no-op if absent)."""
    TEMPLATE_REGISTRY.pop(name, None)


def get_registered_templates() -> list[Pass]:
    """Return registered templates in insertion order (Python 3.7+ dicts preserve insertion order)."""
    return list(TEMPLATE_REGISTRY.values())
