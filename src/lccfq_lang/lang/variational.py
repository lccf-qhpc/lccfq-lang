"""Variational ansatz primitives: hardware-efficient ansatz and QAOA step."""

from typing import List

from ..arch.instruction import Instruction
from ..arch.isa import ISA


def hw_eff_ansatz(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Apply a hardware-efficient ansatz layer to target qubits.

    :param isa: instruction set architecture
    :param target: list of qubit indices
    :return: list of instructions implementing the hardware-efficient ansatz
    """
    pass


def qaoa_step(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Apply a single QAOA step to target qubits.

    :param isa: instruction set architecture
    :param target: list of qubit indices
    :return: list of instructions implementing a QAOA step
    """
    pass
