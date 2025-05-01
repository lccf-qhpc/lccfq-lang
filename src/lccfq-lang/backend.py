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
from dataclasses import dataclass
from typing import List
from .topology import Topology


class QPUStatus(Enum):
    """Possible states the QPU can be in.

    Note that numbering is ordered in terms of best possible state (idle) to run programs
    and error states are worse the more negative these are.
    """
    INITIALIZED = 1
    TUNING = 2
    BUSY = 3
    IDLE = 4
    BAD_TUNING = -1
    UNRESPONSIVE = -2
    NO_ANSWER = -3


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
    topology: Topology
    connection: QPUConnection


class QPU:
    """A `QPU` determines all interactions with the device through issuing circuit, control and benchmark
    instructions. Accessing the backend requires submitting requests through a REST interface. An HPC system
    will run a single instance of the backend, which will communicate with lccfq-lang through this interface.

    New programs will always import this library.
    """

    def __init__(self,
                filename: str=None
                 ):
        #Load the configuration and establish the bridge
        self.config = self.__from_file(filename)
        self.__bridge()

        # Use the
        self.transpiler = None #TODO: have a selector based on native gate set

    def __from_file(self,
                    filename: str) -> QPUConfig:
        data = toml.load(filename)

        qpu_data = data["qpu"]
        topology_data = data["topology"]
        network_data = data["network"]

        topology = Topology(
            name=topology_data["name"],
            qubits=topology_data["qubits"],
            connections=topology_data["connections"]
        )

        connection = QPUConnection(
            ip=network_data["ip"],
            port=network_data["port"]
        )

        return QPUConfig(
            name=qpu_data["name"],
            location=qpu_data["location"],
            qubit_count=qpu_data["qubit_count"],
            native_gates=qpu_data["native_gateset"],
            topology=topology,
            connection=connection
        )

    def __bridge(self):
        """
        Use ZMQ to connect this program instance to a specific communication queue.
        :return: Nothing
        """
        pass