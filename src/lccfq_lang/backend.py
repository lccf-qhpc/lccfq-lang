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
from lccfq_lang.defaults import Mach
from lccfq_lang.mach.ir import Gate, Control
from lccfq_lang.mach.topology import QPUTopology
from lccfq_lang.mach.transpilers import TranspilerFactory
from lccfq_lang.arch.preconds import Precondition
from lccfq_lang.arch.postconds import Postcondition
from lccfq_lang.arch.instruction import Instruction


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
    qubits: List[int]
    exclusions: List[int]
    topology: QPUTopology
    connection: QPUConnection


class QPU:
    """A `QPU` determines all interactions with the device through issuing circuit, control and benchmark
    instructions. Accessing the backend requires submitting requests through a REST interface. An HPC system
    will run a single instance of the backend, which will communicate with lccfq_lang through this interface.

    New programs will always import this library.
    """

    def __init__(self,
                filename: str=None
                ):
        #Load the configuration and establish the bridge
        self.config = self.__from_file(filename)
        self.__bridge()

        # Check that we at least have a default transpiler
        if self.config.name is None:
            self.transpiler = Mach.transpiler
        else:
            self.transpiler = TranspilerFactory().get(self.config.name)

    @staticmethod
    def __from_file(filename: str) -> QPUConfig:
        data = toml.load(filename)

        qpu_data = data["qpu"]
        topology_data = data["topology"]
        network_data = data["network"]

        topology = QPUTopology(
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
            qubits=qpu_data["qubits"],
            exclusions=qpu_data["exclusions"],
            topology=topology,
            connection=connection
        )

    def __bridge(self):
        """
        Use ZMQ to connect this program instance to a specific communication queue.
        :return: Nothing
        """
        pass

    def __check_precon(self, precon: Precondition) -> bool:
        """Check whether a precondition is met

        :param precon: precondition to meet prior to instruction execution
        :return: validity of precondition under current QPU state
        """
        pass

    def __check_postcon(self, precon: Postcondition) -> bool:
        """Check whether a precondition is met

        :param postcon: postcondition the QPU must meet to consider validity
               of execution
        :return: validity of postcondition after QPU state altered by instruction
        """
        pass

    def exec_single(self, instruction: Instruction):
        """Execute a single instruction.
        :param instruction: instruction to execute
        :return: Nothing"""
        pass

    def exec_circuit(self, program: List[Gate|Control]):
        """
        Execute the result of transpiling a circuit.

        :param program: a program resulting from a quantum circuit, already transpiled
        :return: nothing
        """
        pass
