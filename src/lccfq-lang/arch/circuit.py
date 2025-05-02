"""
Filename: circuit.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file provides the definition for a quantum circuit as a sequence
    of gates on a number of qubits. Circuits utilize a number of qubits
    with no assumption about the mapping, which is provided by the machine
    model (`mach`).

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from .instruction import Instruction
from typing import List


class Circuit:
    """Implementation of a quantum circuit based on instructions.
    """

    def __init__(self,
                 shots: int = 1000):
        self.shots = shots
        self.operations: List[Instruction] = list()

    def add(self,
            instruction: Instruction):
        """Add an instruction to the circuit.

        :param instruction: The instruction to add to the circuit.
        :return: Nothing.
        """
        self.operations.append(instruction)
