"""
Filename: register.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file defines a quantum register in LCCFQ.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import copy
import numpy as np

from enum import Enum
from typing import List, Dict
from .error import NoMeasurementsAvailable, MalformedInstruction, NotAllowedInContext
from .instruction import Instruction, InstructionType
from .isa import ISA
from ..backend import QPU


class QContext(Enum):
    """
    A context provides information about constraints that instructions must comply with.

    """
    CIRCUIT = 0
    TEST = 1
    CONTROL = 2
    UNKNOWN = -1


class QRegister:
    """
    Class that manages the definition of a quantum register.

    The role of registers in this programming models is to serve as
    a representation that can
    """

    def __init__(self, qubit_count: int, qpu: QPU) -> None:
        """Creates a register with qubits as abstract lines and a mapping to real hardware

        :param qubit_count:
        :param mapping:
        """
        self.qubit_count = qubit_count
        self.qpu = qpu

    def map(self, instruction: Instruction) -> Instruction:
        """Forward the mapping of an instruction provided by the QPU.

        :param instruction: original instruction
        :return: mapped instruction
        """
        return self.qpu.map(instruction)

    def swaps(self, instruction: Instruction, isa: ISA) -> List[Instruction]:
        """
        Forward adding swaps from the mapping and its topology.

        :param instruction: instruction without swaps
        :param isa: instruction set architecture
        :return: list of instructions with potential swaps
        """
        return self.qpu.mapping.swaps(instruction, isa)

    def expand(self, instruction: Instruction) -> List[Instruction]:
        """
        Apply an instruction to the register. We obtain a list of new instructions before
        performing swaps on the gates. We assume an instruction has already been challenged.

        :param instruction: instruction to apply
        :return: expanded instruction list
        """
        # Step 1: map the instruction from virtual to physical qubits
        mapped_instruction = self.qpu.mapping.map(instruction)

        # Step 2: we have instructions that must be themselves expanded before SWAPS are introduced

        ## Case 1: u2(phi, lambda)
        if instruction.symbol == "u2":
            phi = instruction.parameters[0]
            lbmd = instruction.parameters[1]

            return [
                self.qpu.isa.rz(tg=instruction.target_qubits[0], params=[phi]),
                self.qpu.isa.ry(tg=instruction.target_qubits[0], params=[np.pi/2]),
                self.qpu.isa.rz(tg=instruction.target_qubits[0], params=[lbmd])
            ]
        ## Case 2: u3
        elif instruction.symbol == "u3":
            phi = instruction.parameters[0]
            theta = instruction.parameters[1]
            lbmd = instruction.parameters[2]

            return [
                self.qpu.isa.rz(tg=instruction.target_qubits[0], params=[phi]),
                self.qpu.isa.ry(tg=instruction.target_qubits[0], params=[theta]),
                self.qpu.isa.rz(tg=instruction.target_qubits[0], params=[lbmd])
            ]
        ## Case 3: cu
        elif instruction.symbol == "cu":
            phi = instruction.parameters[0]
            theta = instruction.parameters[1]
            lbmd = instruction.parameters[2]

            return [
                self.qpu.isa.rz(tg=instruction.target_qubits[0], params=[lbmd]),
                self.qpu.isa.ry(tg=instruction.target_qubits[0], params=[theta/2]),
                self.qpu.isa.cx(ct=instruction.control_qubits[0], tg=instruction.target_qubits[0]),
                self.qpu.isa.ry(tg=instruction.target_qubits[0], params=[-theta/2]),
                self.qpu.isa.rz(tg=instruction.target_qubits[0], params=[-(phi + lbmd)]),
                self.qpu.isa.cx(ct=instruction.control_qubits[0], tg=instruction.target_qubits[0]),
                self.qpu.isa.rz(tg=instruction.target_qubits[0], params=[phi])
            ]
        ## Case 4: cu
        elif instruction.symbol == "measure" and len(instruction.target_qubits) > 1:
            return [
                self.qpu.isa.measure(tgs=[q]) for q in instruction.target_qubits
            ]
        else:
            return [instruction]

    def challenge(self, instruction: Instruction, context: QContext) -> Instruction:
        """Ensure an instruction is valid and well-formed. Errors are raised
        as exceptions. Challenging is performed for a specific QPU.
    
        :param instruction: instruction to challenge
        :param context: context under which we interpret the challenge
        :return: modified instruction after challenge
        """

        # Context-free challenge: is the instruction well-formed?
        _ = self._is_well_formed_instruction(instruction)

        # Create a new deep copy of the instruction
        instr = copy.deepcopy(instruction)

        # Case 1: no control or test instructions while executing a circuit
        if context == QContext.CIRCUIT:
            if instruction.instruction_type in [InstructionType.QPUSTATE, InstructionType.TEST]:
                raise NotAllowedInContext(instruction, context)

            # We are in a circuit, remove shot data
            instr.instruction_type = InstructionType.CIRCUIT
            instr.shots = None
        # Case 2: no QPU control instructions when executing a test block
        elif context == QContext.TEST:
            if instruction.instruction_type == InstructionType.QPUSTATE:
                raise NotAllowedInContext(instruction, context)

            if instruction.shots is None:
                raise MalformedInstruction(instruction, "tests must indicate number of shots")

            # Note that a gate, when interpreted as a test, will be executed and return a measurement
            # automatically
            instr.instruction_type = InstructionType.TEST
        else:
            # We have a general QPU control instruction which will occur outside of a context
            instr.instruction_type = InstructionType.QPUSTATE

        return instr

    def all(self):
        return self.qpu.mapping.virtual_qubits

    def but(self, minus: List[int]=None):
        if minus is None:
            return self.qpu.mapping.virtual_qubits
        else:
            return list(set(self.qpu.mapping.virtual_qubits) - set(minus))

    @staticmethod
    def _is_well_formed_instruction(instruction: Instruction) -> bool:
        """Determine if the instruction is well-formed.

        :param instruction: instruction to test
        :return: True if valid, raises ValueError otherwise
        """
        if not isinstance(instruction.symbol, str) or not instruction.symbol:
            raise MalformedInstruction(instruction, "symbol must be a non-empty string")

        if not isinstance(instruction.target_qubits, list) or not instruction.target_qubits:
            if instruction.instruction_type != InstructionType.QPUSTATE:
                raise MalformedInstruction(instruction, "target qubits must be a non-empty list")

        if instruction.target_qubits is not None:
            if not all(isinstance(q, int) and q >= 0 for q in instruction.target_qubits):
                raise MalformedInstruction(instruction, "target qubits must be non-negative integers")

        if instruction.is_controlled:
            if not isinstance(instruction.control_qubits, list) or not instruction.control_qubits:
                raise MalformedInstruction(instruction, "control qubits must be present if controlled")

            if not all(isinstance(q, int) and q >= 0 for q in instruction.control_qubits):
                raise MalformedInstruction(instruction, "control qubits must be non-negative integers")

            if set(instruction.control_qubits).intersection(instruction.target_qubits):
                raise MalformedInstruction(instruction, "target and control qubits must be different")

        if instruction.parameters is not None:
            if not isinstance(instruction.parameters, list):
                raise MalformedInstruction(instruction, "parameters must be a list of real values")
            if not all(isinstance(p, float) for p in instruction.parameters):
                raise MalformedInstruction(instruction, "all parameters must be real values")

        if instruction.shots is not None:
            if not isinstance(instruction.shots, int) or instruction.shots <= 0:
                raise MalformedInstruction(instruction, "shot count must be positive integer")

        return True


class CRegister:
    """A classical register implementation. Internally, a register should
    operate as an ensemble obtained from a measurement.
    """

    def __init__(self, size: int):
        self.bit_count = size
        self.data = None

    def absorb(self, data: Dict[str, int]):
        self.data = data

    def frequencies(self):
        if self.data is None:
            raise NoMeasurementsAvailable()

        total = sum(self.data.values())

        if total == 0:
            return {k: 0.0 for k in self.data}

        return {k: v / total for k, v in self.data.items()}
