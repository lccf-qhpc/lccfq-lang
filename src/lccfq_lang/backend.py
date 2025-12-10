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
import grpc
import toml

from enum import Enum
from typing import List, Dict
from .defaults import Mach
from .mach.ir import Gate, Control, Test
from .mach.topology import QPUTopology
from .arch.isa import ISA
from .arch.mapping import QPUMapping
from .arch.register import QRegister
from .arch.preconds import Precondition
from .arch.postconds import Postcondition
from .arch.error import UnknownCompilerPass
from .sys.error import BadQPUConfiguration
from .arch.instruction import Instruction
from .sys.base import QPUConfig
from .sys.factories.mach import TranspilerFactory

from lccfq_backend.utils.log import setup_logger
from lccfq_backend.api.client import Client

logger = setup_logger("lccfq_lang.QPU")


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
    instructions. Accessing the backend requires submitting requests through a gRPC interface. An HPC system
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
        # If last_pass is None or unrecognized, default to "transpiled"
        if last_pass not in [
            "parsed",
            "mapped",
            "swapped",
            "expanded",
            "transpiled",
            "executed"
        ]:
            self.last_pass = last_pass if last_pass is not None else "transpiled"

        # Instantiate the LCCF Instruction Set Architecture
        self.isa = ISA("lccfq")

        # Check that we at least have a default transpiler
        if self.config.name is None:
            self.transpiler = Mach.transpiler
        else:
            self.transpiler = TranspilerFactory().get(self.config.name)

        self.backend_client = None

        # Check if last_pass is executed, in which case ping the backend to test connection.
        if self.last_pass == "executed":
            logger.debug("Last pass is 'executed', setting up connection with backend.")
            con = self.config.connection
            self.backend_client = Client(name=con.username,
                                         address=con.address,
                                         port=con.port,
                                         clients_cert_dir=con.client_cert_dir,
                                         server_cert_dir=con.server_cert)
            logger.info("Pinging QPU backend to check connectivity...")
            try:
                if not self.backend_client.ping():
                    raise ConnectionError("Could not connect to QPU backend.")
            except grpc.RpcError or ConnectionError as e:
                logger.error("Could not connect to QPU backend.")
                raise e

            logger.info("Successfully connected to QPU backend.")

    @staticmethod
    def __from_file(filename: str) -> QPUConfig:
        try:
            data = toml.load(filename)
        except FileNotFoundError:
            raise BadQPUConfiguration("valid config file", f"file not found: {filename}")
        except toml.TomlDecodeError as e:
            raise BadQPUConfiguration("valid TOML", f"parse error in {filename}: {e}")

        return QPUConfig(data)

    def __bridge(self):
        """
        Use gRPC to connect this program instance to a specific HPC backend.
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

    def qregister(self, qubit_count: int) -> QRegister:
        """Create a QRegister bound to this QPU's mapping and ISA.

        :param qubit_count: number of qubits
        :return: a new QRegister
        """
        return QRegister(qubit_count, self.mapping, self.isa)

    def exec_single(self, instruction: Instruction, shots: int):
        """Execute a single instruction.
        :param instruction: instruction to execute
        :param shots: number of shots
        :return: Nothing"""
        pass

    def exec_circuit(self, circuit: List[Gate|Test|Control], shots: int) -> Dict[str, float]:
        """
        Execute the result of transpiling a circuit.

        :param circuit: a program resulting from a quantum circuit, already transpiled
        :param shots: number of shots
        :return: the results count from executing a circuit
        """
        response = self.backend_client.submit_circuit_task(circuit, shots)
        return {}

    def map(self, instruction: Instruction) -> Instruction:
        """Forward the mapping of an instruction provided by the internal mapping.

        :param instruction: original instruction
        :return: mapped instruction
        """
        return self.mapping.map(instruction)
