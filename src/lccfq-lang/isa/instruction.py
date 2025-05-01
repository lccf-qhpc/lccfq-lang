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

class InstructionType(Enum):
    """InstructionType describes the main classes of instructions in LCCF code
    users will issue to a QPU.
    """
    CIRCUIT = 1
    STATE = 2
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
                 has_effects: bool = True
                 ):
        # Basis properties of an instruction
        self.symbol = symbol
        self.instruction_type = instruction_type
        self.is_native = is_native
        self.has_effects = has_effects

        # Pre- and post-conditions of the hoare triplet
        self.pre: Set[Precondition] = set()
        self.post: Set[Postcondition] = set()

    def transpile(self, qpu: QPU):
        """
        Transpile a
        :return:
        """
        pass