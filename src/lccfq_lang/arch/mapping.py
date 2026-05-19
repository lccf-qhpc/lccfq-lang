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

    _VALID_ROUTING_STRATEGIES: tuple = ("identity", "sabre_lite", "sabre_fast")

    def __init__(
        self,
        virtual_qubits: List[int],
        topology: QPUTopology,
        routing_strategy: str = "identity",
    ) -> None:
        """During initialization, we will provide simple mappings. Mappings may be relabeled
        as needed to facilitate primitive forms of quantum compilation, but optimization will
        be limited.

        :param virtual_qubits: list of virtual qubits to map
        :param topology: Physical topology of the QPU.
        :param routing_strategy: one of "identity" (default, legacy greedy) or "sabre_lite".
        """
        if routing_strategy not in self._VALID_ROUTING_STRATEGIES:
            raise ValueError(
                f"QPUMapping: routing_strategy must be one of "
                f"{self._VALID_ROUTING_STRATEGIES}, got {routing_strategy!r}"
            )

        self.virtual_qubits = virtual_qubits
        self.topology = topology
        self.routing_strategy = routing_strategy

        if len(self.virtual_qubits) > len(self.topology.qubits()):
            raise NotEnoughQubits(len(self.virtual_qubits), len(self.topology.qubits()))

        self.mapping = {
            v: p for v, p in zip(self.virtual_qubits, self.topology.qubits())
        }

    def with_layout(self, new_layout: dict) -> "QPUMapping":
        """Return a new QPUMapping with the supplied virtual->physical mapping.

        Does not mutate self. Preserves topology, virtual_qubits ordering, and
        routing_strategy.

        :param new_layout: dict mapping virtual qubit id -> physical qubit id
        :return: new QPUMapping with the given layout
        :raises ValueError: if keys/values are invalid
        """
        if set(new_layout.keys()) != set(self.virtual_qubits):
            raise ValueError(
                "QPUMapping.with_layout: new_layout keys must equal virtual_qubits"
            )
        physical = list(new_layout.values())
        if len(set(physical)) != len(physical):
            raise ValueError(
                "QPUMapping.with_layout: physical assignments must be unique"
            )
        if not set(physical).issubset(set(self.topology.qubits())):
            raise ValueError(
                "QPUMapping.with_layout: physical values must be topology qubits"
            )
        clone = QPUMapping(
            virtual_qubits=list(self.virtual_qubits),
            topology=self.topology,
            routing_strategy=self.routing_strategy,
        )
        clone.mapping = dict(new_layout)
        return clone

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
            params=instruction.params,
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
