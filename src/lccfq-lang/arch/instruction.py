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
from ..backend import QPU
from typing import List


class InstructionType(Enum):
    """InstructionType describes the main classes of instructions in LCCF code
    users will issue to a QPU.
    """
    CIRCUIT = 1
    CONTROL = 2
    BENCHMARK = 3


class Instruction(ABC):
    """An Instruction models a mnemonic entity that has executable consequences in
    hardware connected to a QPU. Some instructions may not have a direct executable
    effect, but modulate the execution of other instructions.
    """
    def __init__(self,
                 symbol: str,
                 instruction_type:InstructionType,
                 is_native: bool = False,
                 has_effects: bool = True,
                 is_controlled: bool = False,
                 target_qubits: List[int] = None,
                 control_qubits: List[int] = None,
                 parameters: List[float] = None,
                 shots: int = None,
                 ):
        # Basis properties of an instruction
        self.symbol = symbol
        self.instruction_type = instruction_type
        self.is_native = is_native
        self.has_effects = has_effects
        self.is_controlled = is_controlled
        self.target_qubits = target_qubits
        self.control_qubits = control_qubits
        self.parameters = parameters
        self.shots = shots

        # Pre- and post-conditions of the hoare triplet
        self.pre: Set[Precondition] = set()
        self.post: Set[Postcondition] = set()

    def add_precondition(self,
                         precondition: Precondition) -> None:
        """Add a callable precondition to this instruction.

        :param precondition:
        :return:
        """
        self.pre.add(precondition)
