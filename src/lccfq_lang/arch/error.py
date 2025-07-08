"""
Filename: error.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file defines possible architecture error types in LCCFQ.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""


class NotEnoughQubits(Exception):
    """Exception raised when the QPU topology not the correct one.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, expected, present):
        self.message = f"Not enough qubits available - expected: {expected} \tpresent: {present}"
        super().__init__(self.message)

class NoMeasurementsAvailable(Exception):
    """Exception raised when the QPU topology not the correct one.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self):
        self.message = f"No measurements available yet in the current classical register"
        super().__init__(self.message)


class BadParameterCount(Exception):
    """Exception raised when an instruction does not fulfill its parameter count.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, expected, present):
        self.message = f"Bad parameter count - expected: {expected}\tpresent: {present}"
        super().__init__(self.message)


class UndefinedParametricInstruction(Exception):
    """Exception raised when an instruction does not fulfill its parameter count.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, instruction, expected, present):
        self.message = f"No instruction with given parameters - inst: {instruction}\texpected: {expected}\tpresent: {present}"
        super().__init__(self.message)
