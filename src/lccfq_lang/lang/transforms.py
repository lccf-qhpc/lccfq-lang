"""Unitary transforms: quantum Fourier transform and its inverse."""

from typing import List

from ..arch.instruction import Instruction
from ..arch.isa import ISA


def qft(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Apply the quantum Fourier transform to target qubits.

    :param isa: instruction set architecture
    :param target: list of qubit indices
    :return: list of instructions implementing QFT
    """
    pass


def iqft(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Apply the inverse quantum Fourier transform to target qubits.

    :param isa: instruction set architecture
    :param target: list of qubit indices
    :return: list of instructions implementing inverse QFT
    """
    pass
