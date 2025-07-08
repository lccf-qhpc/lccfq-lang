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
from typing import List, Dict
from .error import NoMeasurementsAvailable
from ..backend import QPU
from .instruction import Instruction


class QRegister:
    """
    Class that manages the definition of a quantum register.

    The role of registers in this programming models is to serve as
    a representation that can
    """

    def __init__(self, qubit_count: int, qpu: QPU) -> None:
        """
        Creates a register with qubits as abstract lines and a mapping to real hardware

        :param qubit_count:
        :param mapping:
        """
        self.qubit_count = qubit_count
        self.qpu = qpu

    def expand(self, instruction: Instruction) -> List[Instruction]:
        """
        Apply an instruction to the register. We obtain a list of new instructions based on the potential
        need to perform swaps on the gates. We assume an instruction has already been challenged.

        :param instruction: instruction to apply
        :return: expanded instruction list
        """
        return []

    def challenge(self, instruction: Instruction):
        """Ensure an instruction is valid and well-formed. Errors are raised
        as exceptions. Challenging is performed for a specific QPU.
    
        :param instruction: instruction to test
        :return: nothing
        """
        pass


class CRegister:
    """A classical register implementation. Internally, a register should
    operate as an ensemble obtained from a measurement.
    """

    def __init__(self, size: int):
        self.bit_count = size
        self.data = None

    def absorb(self, data: Dict[str, int]):
        self.data = data

    def frequencies(self):
        if self.data is None:
            raise NoMeasurementsAvailable()

        total = sum(self.data.values())

        if total == 0:
            return {k: 0.0 for k in self.data}

        return {k: v / total for k, v in self.data.items()}
