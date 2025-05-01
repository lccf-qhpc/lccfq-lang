"""
Filename: topology.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file defines qubit topologies required to transpile ideal instructions
    into native gate sets.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""

from dataclasses import dataclass
from typing import List, Set


@dataclass
class Topology:
    name: str
    qubits: Set[int]
    connections: List[List[int]]
