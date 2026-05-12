"""Amplitude amplification primitives."""

from typing import List

from ..arch.instruction import Instruction
from ..arch.isa import ISA


def diffusion(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Apply the Grover diffusion operator to target qubits.

    :param isa: instruction set architecture
    :param target: list of qubit indices
    :return: list of instructions implementing the diffusion operator
    """
    pass
