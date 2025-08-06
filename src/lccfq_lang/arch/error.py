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


class MalformedInstruction(Exception):
    """Exception raised when an instruction is not well-formed.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, instruction, cause):
        self.message = f"Malformed instruction - inst: {instruction}\tcause: {cause}"
        super().__init__(self.message)


class UnknownInstruction(Exception):
    """Exception raised when an instruction is not found in the ISA. We
    define this to detect possible code injections in the future.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, instruction):
        self.message = f"Unrecognized instruction - inst: {instruction}"
        super().__init__(self.message)


class UnknownCompilerPass(Exception):
    """Exception raised when an unknown compiler pass is found.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, cpass):
        self.message = f"Unrecognized compiler pass - pass: {cpass}"
        super().__init__(self.message)


class NotAllowedInContext(Exception):
    """Exception raised when an instruction is not allowed within a specific context.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, instruction, context):
        self.message = f"Context prevents {instruction} in: {context}"
        super().__init__(self.message)

class BadQPUConfiguration(Exception):
    """Exception raised when the QPU topology not the correct one.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, expected, present):
        self.message = f"QPU misconfigured - expected: {expected} \tpresent: {present}"
        super().__init__(self.message)
