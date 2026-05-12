"""Quantum arithmetic primitives: addition, modular multiplication, comparison."""

from typing import List

from ..arch.instruction import Instruction
from ..arch.isa import ISA


def add(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Add two quantum registers.

    :param isa: instruction set architecture
    :param target: list of qubit indices
    :return: list of instructions implementing quantum addition
    """
    pass


def mult_mod(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Multiply two quantum registers modulo a classical constant.

    :param isa: instruction set architecture
    :param target: list of qubit indices
    :return: list of instructions implementing modular multiplication
    """
    pass


def compare(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Compare two quantum registers.

    :param isa: instruction set architecture
    :param target: list of qubit indices
    :return: list of instructions implementing quantum comparison
    """
    pass
