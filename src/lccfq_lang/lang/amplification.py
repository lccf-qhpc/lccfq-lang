"""Amplitude amplification primitives."""

from typing import List

from ..arch.instruction import Instruction
from ..arch.isa import ISA
from ._common import _mcz


def diffusion(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Grover diffusion operator D = 2|s><s| - I where |s> is the uniform
    superposition over ``target``.

    Textbook decomposition: H -> X -> multi-controlled Z -> X -> H, all
    applied across every qubit in target. The MCZ implements the phase
    flip on |11..1>; conjugation by X then by H rotates that into the
    "flip about |s>" reflection.

    :param isa: instruction set architecture
    :param target: list of qubit indices (length n >= 1)
    :return: list of instructions implementing the diffusion operator
    """
    n = len(target)

    if n < 1:
        raise ValueError("diffusion requires at least 1 qubit")

    instructions = []

    for q in target:
        instructions.append(isa.h(tg=q))
    for q in target:
        instructions.append(isa.x(tg=q))

    instructions.append(_mcz(list(target)))

    for q in target:
        instructions.append(isa.x(tg=q))
    for q in target:
        instructions.append(isa.h(tg=q))

    return instructions
