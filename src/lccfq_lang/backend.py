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
from typing import List, Dict
from .defaults import Mach
from .mach.ir import Gate, Control
from .mach.topology import QPUTopology
from .arch.isa import ISA
from .arch.mapping import QPUMapping
from .arch.preconds import Precondition
from .arch.postconds import Postcondition
from .arch.instruction import Instruction
from .sys.base import QPUConfig
from .sys.factories.mach import TranspilerFactory


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


class QPU:
    """A `QPU` determines all interactions with the device through issuing circuit, control and benchmark
    instructions. Accessing the backend requires submitting requests through a REST interface. An HPC system
    will run a single instance of the backend, which will communicate with lccfq_lang through this interface.

    New programs will always import this library.
    """

    def __init__(self,
                filename: str=None,
                last_pass: str=None
                ):
        #Load the configuration and establish the bridge
        self.config = self.__from_file(filename)
        virtual_qubits = list(range(self.config.qubit_count))
        self.mapping = QPUMapping(virtual_qubits, QPUTopology(self.config))
        self.last_pass = last_pass
        self.__bridge()


        # Set last compilation/transpilation that produces code
        if not last_pass in [
            "dryrun",
            "mapped",
            "swaps",
            "expanded",
            "transpiled",
            "executed"
        ]:
            pass

        # Instantiate the LCCF Instruction Set Architecture
        self.isa = ISA("lccfq")

        # Check that we at least have a default transpiler
        if self.config.name is None:
            self.transpiler = Mach.transpiler
        else:
            self.transpiler = TranspilerFactory().get(self.config.name)

    @staticmethod
    def __from_file(filename: str) -> QPUConfig:
        data = toml.load(filename)

        return QPUConfig(data)

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

    def exec_circuit(self, circuit: List[Gate|Control]) -> Dict[str, int]:
        """
        Execute the result of transpiling a circuit.

        :param circuit: a program resulting from a quantum circuit, already transpiled
        :return: the results count from executing a circuit
        """
        return {}

    def map(self, instruction: Instruction) -> Instruction:
        """Forward the mapping of an instruction provided by the internal mapping.

        :param instruction: original instruction
        :return: mapped instruction
        """
        return self.mapping.map(instruction)
