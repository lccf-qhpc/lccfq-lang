"""
Filename: instruction.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file provides classes and enums to define instructions in the
    LCCF QPU language.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Set
from .preconds import Precondition
from .postconds import Postcondition
from typing import List


class InstructionType(Enum):
    """InstructionType describes the main classes of instructions in LCCF code
    users will issue to a QPU.

    An instruction type being delayed means that its use will be determined by its
    context.
    """
    DELAYED = 0
    CIRCUIT = 1
    CONTROL = 2
    BENCHMARK = 3


class Instruction(ABC):
    """An Instruction models a mnemonic entity that has executable consequences in
    hardware connected to a QPU. Some instructions may not have a direct executable
    effect, but modulate the execution of other instructions. Instructions are
    delayed by default.
    """
    def __init__(self,
                 symbol: str,
                 is_native: bool = False,
                 modifies_state: bool = True,
                 is_controlled: bool = False,
                 target_qubits: List[int] = None,
                 control_qubits: List[int] = None,
                 parameters: List[float] = None,
                 shots: int = None,
                 ):
        # Basis properties of an instruction
        self.symbol = symbol
        self.instruction_type = InstructionType.DELAYED
        self.is_native = is_native
        self.modifies_state = modifies_state
        self.is_controlled = is_controlled
        self.is_mapped = False
        self.target_qubits = target_qubits
        self.control_qubits = control_qubits
        self.parameters = parameters
        self.shots = shots

        # Pre- and post-conditions of the hoare triplet
        self.pre: Set[Precondition] = set()
        self.post: Set[Postcondition] = set()

    def __repr__(self):
        return f"{self.symbol} over {self.target_qubits} controlled by {self.control_qubits}"

    def add_precondition(self,
                         precondition: Precondition) -> None:
        """Add a callable precondition to this instruction.

        :param precondition:
        :return:
        """
        self.pre.add(precondition)

    def add_postcondition(self,
                         postcondition: Postcondition) -> None:
        """Add a callable precondition to this instruction.

        :param postcondition:
        :return:
        """
        self.post.add(postcondition)
