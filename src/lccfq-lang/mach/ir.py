"""
Filename: ir.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file provides entities involved in the LCCFQ intermediate representation

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from dataclasses import dataclass
from typing import List


@dataclass
class Gate:
    """
    Gates assume that the ordering of application in the circuit is the same as the
    diagram of that circuit.
    """
    name: str
    target_qubits: List[int]
    control_qubits: List[int]
