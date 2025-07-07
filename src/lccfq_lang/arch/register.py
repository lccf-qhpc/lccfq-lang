"""
Filename: register.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file defines a quantum register in LCCFQ.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from typing import List
from .mapping import QPUMapping
from .instruction import Instruction


class QRegister:
    """
    Class that manages the definition of a quantum register.

    The role of registers in this programming models is to serve as
    a representation that can
    """

    def __init__(self, qubit_count: int, mapping: QPUMapping) -> None:
        """
        Creates a register with qubits as abstract lines and a mapping to real hardware

        :param qubit_count:
        :param mapping:
        """
        self.qubit_count = qubit_count
        self.mapping = mapping

    def apply(self, instruction: Instruction) -> List[Instruction]:
        """
        Apply an instruction to the register.

        :param instruction: instruction to apply
        :return: expanded instruction list
        """
        return self.mapping.map(instruction)
