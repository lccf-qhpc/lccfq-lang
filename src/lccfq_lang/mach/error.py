"""
Filename: error.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file defines possible device error types in LCCFQ.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""


class BadTopologyType(Exception):
    """Exception raised when the QPU topology not the correct one.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, expected):
        self.message = f"Machine topology different from specified - expected: {expected}"
        super().__init__(self.message)

class InsufficientGoodQubits(Exception):
    """Exception raised when the QPU lacks enough good qubits either by static definition
    or during execution.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, expected, actual):
        self.message = f"Insufficient number of good qubits to satisfy request - expected: {expected}\tactual: {actual}"
        super().__init__(self.message)


class QubitsNotConnected(Exception):
    """Exception raised when the QPU topology lacks connections between two qubits.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, qa, qb):
        self.message = f"Qubits not physically connected - qa: {qa} \tpresent: {qb}"
        super().__init__(self.message)
