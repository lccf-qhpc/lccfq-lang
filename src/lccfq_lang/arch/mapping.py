"""
Filename: mapping.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file defines a mapping from virtual to real qubits in LCCFQ.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from .instruction import Instruction, InstructionType
from .isa import ISA
from .error import NotEnoughQubits
from ..mach.topology import QPUTopology
from typing import List


class QPUMapping:
    """
    A mapping from virtual to real qubits in LCCFQ.
    """

    def __init__(self, virtual_qubits: List[int], topology: QPUTopology) -> None:
        """During initialization, we will provide simple mappings. Mappings may be relabeled
        as needed to facilitate primitive forms of quantum compilation, but optimization will
        be limited.

        :param virtual_qubits: list of virtual qubits to map
        :param topology: Physical topology of the QPU.
        """
        self.virtual_qubits = virtual_qubits
        self.topology = topology

        if len(self.virtual_qubits) > len(self.topology.qubits()):
            raise NotEnoughQubits(len(self.virtual_qubits), len(self.topology.qubits()))

        self.mapping = {
            v: p for v, p in zip(self.virtual_qubits, self.topology.qubits())
        }

    def map(self, instruction: Instruction) -> Instruction:
        """Maps an instruction into the provided topology.

        :param instruction:
        :return: mapped instruction.
        """
        mapped_targets = (
            [self.mapping[q] for q in instruction.target_qubits]
            if instruction.target_qubits
            else []
        )

        mapped_controls = (
            [self.mapping[q] for q in instruction.control_qubits]
            if instruction.control_qubits
            else []
        )

        mapped_instruction = Instruction(
            symbol=instruction.symbol,
            modifies_state=instruction.modifies_state,
            is_controlled=instruction.is_controlled,
            target_qubits=mapped_targets,
            control_qubits=mapped_controls,
            parameters=instruction.parameters,
            shots=instruction.shots
        )

        # We may have a single test using a single gate with many shots
        # or a full circuit
        mapped_instruction.instruction_type = InstructionType.DELAYED
        mapped_instruction.pre = instruction.pre.copy()
        mapped_instruction.post = instruction.post.copy()
        mapped_instruction.is_mapped = True

        return mapped_instruction

    def swaps(self, instruction: Instruction, isa: ISA) -> List[Instruction]:
        """Delegate introducing swaps to the provided topology provided
        instructions are already mapped to physical qubits.

        :param instruction: original instruction
        :param isa: instruction set architecture
        :return: list of instructions possibly couched between swaps
        """
        return self.topology.swaps(instruction, isa)
