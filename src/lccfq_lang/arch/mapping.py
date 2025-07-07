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
from dis import Instruction
from .error import NotEnoughQubits
from ..mach.topology import QPUTopology
from typing import List, Dict


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

        if self.virtual_qubits > self.topology.qubits():
            raise NotEnoughQubits(len(self.virtual_qubits), len(self.topology.qubits()))

        self.map = {}

        # Note that a program *can* request fewer qubits than provided by a device
        for v, p in zip(self.virtual_qubits, self.topology.qubits()):
            self.map[v] = p

    def map(self, instruction: Instruction) -> List[Instruction]:
        """Maps an instruction into a list of instructions based on this mapping.
        SWAP may be introduced for non-contiguous qubits at a high level. Number of
        swaps depends on labeling, as well as potential re-labelings.

        :param instruction:
        :return: list of mapped instructions.
        """
        pass
