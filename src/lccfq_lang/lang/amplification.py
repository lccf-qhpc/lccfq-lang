"""Amplitude amplification primitives."""

from typing import List

from ..arch.instruction import Instruction
from ..arch.isa import ISA
from .multicontrol import mcz


def diffusion(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Grover diffusion operator D = 2|s><s| - I where |s> is the uniform
    superposition over ``target``.

    Textbook decomposition: H -> X -> multi-controlled Z -> X -> H, all
    applied across every qubit in target. The MCZ implements the phase
    flip on |11..1>; conjugation by X then by H rotates that into the
    "flip about |s>" reflection.

    :param isa: instruction set architecture
    :param target: list of qubit indices (length n >= 1)
    :param kwargs:
        workspace: int | None = None
            Optional clean ancilla forwarded to mcz. Required when
            mc_mode="barenco".
        mc_mode: Literal["barenco", "vchain"] = "vchain"
            Decomposition mode for the multi-controlled Z.
    :return: list of instructions implementing the diffusion operator
    """
    n = len(target)

    if n < 1:
        raise ValueError("diffusion requires at least 1 qubit")

    workspace = kwargs.get("workspace", None)
    mc_mode = kwargs.get("mc_mode", "vchain")

    if workspace is not None and workspace in target:
        raise ValueError(
            f"workspace qubit {workspace} must not appear in target {target}"
        )

    instructions = []

    for q in target:
        instructions.append(isa.h(tg=q))
    for q in target:
        instructions.append(isa.x(tg=q))

    instructions.extend(mcz(
        isa,
        list(target[:-1]),
        tg=target[-1],
        mode=mc_mode,
        ancilla=workspace,
    ))

    for q in target:
        instructions.append(isa.x(tg=q))
    for q in target:
        instructions.append(isa.h(tg=q))

    return instructions
