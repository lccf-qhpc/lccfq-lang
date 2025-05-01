"""
Filename: backend.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file defines the backend for the LCCF QPU.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""

import toml
from enum import Enum
from .topology import Topology


class QPUStatus(Enum):
    """Possible states the QPU can be in.
    """


class QPU:
    """A `QPU` determines all interactions with the device through issuing circuit, control and benchmark
    instructions. Accessing the backend requires submitting requests through a REST interface. An HPC system
    will run a single instance of the backend, which will communicate with lccfq-lang through this interface.

    New programs will always import this library.
    """

    def __init__(self,
                 name: str,
                 location: str,
                 qubit_count: int,
                 topology: Topology,
                 ip: str,
                 port: int
                 ):
        pass

    def from_file(self,
                  filename: str):
        pass

