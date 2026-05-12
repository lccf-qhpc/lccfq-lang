"""Qubit movement primitives: swaps and entangling steps."""

from typing import List

from ..arch.instruction import Instruction
from ..arch.isa import ISA


def swap(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Swap two qubits.

    :param isa: instruction set architecture
    :param target: list of qubit indices
    :return: list of instructions implementing the swap
    """
    pass


def entangle_step(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Apply an entangling step to target qubits.

    :param isa: instruction set architecture
    :param target: list of qubit indices
    :return: list of instructions implementing the entangling step
    """
    pass
