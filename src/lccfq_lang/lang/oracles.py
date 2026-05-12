"""Oracle primitives: bit-flip and phase oracles."""

from typing import List

from ..arch.instruction import Instruction
from ..arch.isa import ISA


def oracle(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Apply a bit-flip oracle to target qubits.

    :param isa: instruction set architecture
    :param target: list of qubit indices
    :return: list of instructions implementing the bit-flip oracle
    """
    pass


def phase_oracle(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Apply a phase oracle to target qubits.

    :param isa: instruction set architecture
    :param target: list of qubit indices
    :return: list of instructions implementing the phase oracle
    """
    pass
