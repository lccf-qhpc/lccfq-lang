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
from backend import QPU
from lccfq_lang.arch.instruction import Instruction
from lccfq_lang.arch.register import QRegister, CRegister
from lccfq_lang.mach.ir import Gate, Control
from typing import List


class Circuit:
    """Implementation of a quantum circuit based on instructions.

    A circuit is an atomic program composed of gates and measurements at the
    end. Circuits, after their context closes, generate code that goes into
    a circuit description to the respective backend.
    """

    def __init__(self,
                 qpu: QPU,
                 qreg: QRegister,
                 creg: CRegister,
                 shots: int = 1000):
        """Create a new circuit.

        :param qpu: QPU instance to run this circuit
        :param qreg: quantum register
        :param creg: classical register
        :param shots: number of shot to run this circuit for
        """
        self.qpu = qpu
        self.qreg = qreg
        self.creg = creg
        self.shots = shots
        self.operations: List[Instruction] = list()

    def generate(self) -> List[Gate|Control]:
        """Generate calls the machinery that expands and produces our IR for the LCCFQ
        components. Controls are needed to implement, for instance, circuit blocks to be
        synthesized simultaneously.

        :return: List of gates and controls so far contained in the circuit.
        """
        return []

    def __rshift__(self, instr: Instruction):
        """Add a new instruction to the circuit using the `>>` operator. This
        removes verbosity.

        :param instr: instruction to add
        :return: none
        """
        self.qreg.challenge(instr)
        self.operations.append(instr)

    def __enter__(self):
        """Enter the context

        :return: the circuit itself
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, equivalent to generating the circuit and sending it to the
        backend. This allows multiple circuits to have multiple backends by construction.

        :param exc_type: none
        :param exc_val: none
        :param exc_tb: none
        :return: none
        """
        program = self.generate()
        self.qpu.exec_circuit(program)
