"""
Filename: base.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file defines base, shared specifications.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from dataclasses import dataclass
from typing import List


@dataclass
class QPUConnection:
    ip: str
    port: int


@dataclass
class QPUConfig:
    """Representation of the configuration that a QPU requires to operate inside LCCF.
    """
    name: str
    location: str
    qubit_count: int
    native_gates: List[str]
    qubits: List[int]
    exclusions: List[int]
    connection: QPUConnection
