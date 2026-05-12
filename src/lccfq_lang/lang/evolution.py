"""Hamiltonian evolution primitives: continuous time evolution and Trotter decomposition."""

from typing import List

from ..arch.instruction import Instruction
from ..arch.isa import ISA


def time_evolution(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Apply continuous time evolution under a Hamiltonian to target qubits.

    :param isa: instruction set architecture
    :param target: list of qubit indices
    :return: list of instructions implementing time evolution
    """
    pass


def trotter_steps(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Apply Trotter decomposition steps to target qubits.

    :param isa: instruction set architecture
    :param target: list of qubit indices
    :return: list of instructions implementing Trotter steps
    """
    pass
