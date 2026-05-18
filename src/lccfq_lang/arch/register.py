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

from enum import Enum
from typing import List, Dict
from .error import NoMeasurementsAvailable, MalformedInstruction, NotAllowedInContext
from .instruction import Instruction, InstructionType
from .isa import ISA
from .mapping import QPUMapping


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

    def __init__(self, qubit_count: int, mapping: QPUMapping, isa: ISA) -> None:
        """Creates a register with qubits as abstract lines and a mapping to real hardware

        :param qubit_count:
        :param mapping: QPU mapping from virtual to physical qubits
        :param isa: instruction set architecture
        """
        self.qubit_count = qubit_count
        self.mapping = mapping
        self.isa = isa

    def map(self, instruction: Instruction) -> Instruction:
        """Forward the mapping of an instruction provided by the QPU.

        :param instruction: original instruction
        :return: mapped instruction
        """
        return self.mapping.map(instruction)

    def swaps(self, instruction: Instruction, isa: ISA) -> List[Instruction]:
        """
        Forward adding swaps from the mapping and its topology.

        :param instruction: instruction without swaps
        :param isa: instruction set architecture
        :return: list of instructions with potential swaps
        """
        return self.mapping.swaps(instruction, isa)

    def expand(self, instruction: Instruction) -> List[Instruction]:
        """Expand a single instruction into its primitive sequence.

        .. deprecated:: Phase 1
            Direct calls to QRegister.expand are deprecated. Decomposition is
            now performed by the lower_expand PassGroup. This shim will be
            removed in a future release.
        """
        # Local import: avoids circular dependency between arch.register and opt.builtin.
        from lccfq_lang.opt.builtin.lower_universal import (
            LowerU2, LowerU3, LowerCU, FanoutMeasure,
        )
        from lccfq_lang.opt.pass_base import PassContext

        ctx = PassContext(isa=self.isa, mapping=self.mapping)
        program = [instruction]

        for pass_cls in (LowerU2, LowerU3, LowerCU, FanoutMeasure):
            program, _changed = pass_cls(self.isa).run(program, ctx)
        return program

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

    def rebind_mapping(self, new_mapping: "QPUMapping") -> "QRegister":
        """Return a shallow copy of self with .mapping replaced.

        Non-destructive: does not mutate self. Pre-Phase-4 behavior is
        unchanged because callers never invoked this method.

        :param new_mapping: the QPUMapping to bind on the clone
        :return: cloned QRegister with new_mapping
        """
        clone = copy.copy(self)
        clone.mapping = new_mapping
        return clone

    def all(self):
        return self.mapping.virtual_qubits

    def but(self, minus: List[int]=None):
        if minus is None:
            return self.mapping.virtual_qubits
        else:
            return list(set(self.mapping.virtual_qubits) - set(minus))

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

        if instruction.params is not None:
            if not isinstance(instruction.params, list):
                raise MalformedInstruction(instruction, "parameters must be a list of real values")
            if not all(isinstance(p, float) for p in instruction.params):
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
