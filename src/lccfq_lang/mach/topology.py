"""
Filename: topology.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file defines qubit topologies required to transpile ideal instructions
    into native gate sets.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import networkx as nx

from typing import List
from enum import Enum

from ..arch.error import MalformedInstruction
from .error import BadTopologyType, QubitsNotConnected
from ..arch.isa import ISA
from ..arch.instruction import Instruction


class QPUTopoType(Enum):
    """
    Represent possible topologies in a device
    """
    LINEAR = 1
    NONE = -1


class QPUTopology:
    """
    Definition of a specific topology for a QPU
    """

    __type_from_name = {
        "linear": QPUTopoType.LINEAR,
    }

    def __init__(self, topo_spec) -> None:
        self.internal = nx.Graph()

        topo_type = topo_spec["type"]

        if topo_type not in self.__type_from_name.keys():
            raise BadTopologyType(topo_type)

        self.topo_type = self.__type_from_name[topo_type]

        self.real_qubits = self.__remove_exclusions(topo_spec)
        self.real_connections = self.__filter_connections(topo_spec)

        for u, v in self.real_connections:
            self.internal.add_edge(u, v)

        if not self.__test(self.topo_type):
            raise BadTopologyType(self.topo_type)

    def qubits(self) -> List[int]:
        """List of available device qubit indices

        :return: list of qubits
        """
        return list(self.internal.nodes.keys())

    #####################################
    # Tests for different network types #
    #####################################
    def __test_linear(self) -> bool:
        """
        Determine if this topology is linear

        :param topo_type: type to verify
        :return: true if topology is linear
        """
        if self.internal.number_of_nodes() == 0:
            return False

        if not nx.is_connected(self.internal):
            return False

        if self.internal.number_of_edges() != self.internal.number_of_nodes() - 1:
            return False

        degs = [d for n, d in self.internal.degree()]

        return degs.count(1) == 2 and degs.count(2) == self.internal.number_of_nodes() - 2

    def __test(self, topo_type: QPUTopoType) -> bool:
        __dispatch = {
            topo_type.LINEAR: self.__test_linear,
        }

        return __dispatch[topo_type.LINEAR]()


    def __remove_exclusions(self, topo_spec):
        """Calculate the actual usable qubits

        :return: List of usable virtual qubits
        """
        if self.topo_type == "linear":
            # We preserve only elements from the smallest exclusion, since
            # assuming an ordering of qubits numbered from the readout resonator
            # outwards
            min_exclusion = min(topo_spec["exclusions"])
            virtual_qubits = list(filter((lambda q: q < min_exclusion), topo_spec["qubits"]))
        else:
            # For the moment, do a set difference
            virtual_qubits = list(set(topo_spec["qubits"]) - set(topo_spec["exclusions"]))

        return virtual_qubits

    @staticmethod
    def __filter_connections(topo_spec):
        """Filter connections that do not match exclusions in the topology

        :param topo_spec: specification of intended topology
        :return: filtered connections
        """
        if not topo_spec["exclusions"]:
            return topo_spec["connections"]  # nothing to exclude

        min_excluded = min(topo_spec["exclusions"])
        return [c for c in topo_spec["connections"] if min_excluded not in c]

    def swaps(self, instruction: Instruction, isa: ISA) -> List[Instruction]:
        """
        Map an instruction from topology-independent qubits to the specifics of a device
        architecture. The result of a map is a sequence of pairs indicating all swap
        operations required to obtain the results

        :param instruction: original instruction
        :param isa: instruction set architecture for swaps
        :return: list of instructions including potentially sandwiched swaps
        """
        targets = instruction.target_qubits or []
        controls = instruction.control_qubits or []
        all_qubits = controls + targets

        if len(all_qubits) == 1:
            return [instruction]

        if len(all_qubits) != 2:
            raise MalformedInstruction(instruction, f"Unknown {len(all_qubits)}-qubit gate.")

        q0, q1 = all_qubits

        if self.internal.has_edge(q0, q1):
            return [instruction]

        try:
            path = nx.shortest_path(self.internal, source=q0, target=q1)
        except nx.NetworkXNoPath:
            raise QubitsNotConnected(q0, q1)

        pre_swaps = []
        post_swaps = []

        for i in range(len(path) - 2):
            pre_swaps.append(isa.swap(ct=path[i], tg=path[i + 1]))

        routed_q0 = path[-2]
        routed_q1 = path[-1]

        routed_instr = Instruction(
            symbol=instruction.symbol,
            modifies_state=instruction.modifies_state,
            is_controlled=instruction.is_controlled,
            target_qubits=[routed_q1 if q1 in targets else routed_q0],
            control_qubits=[routed_q0 if q0 in controls else routed_q1],
            parameters=instruction.parameters,
            shots=instruction.shots
        )

        for i in reversed(range(len(path) - 2)):
            post_swaps.append(isa.swap(ct=path[i], tg=path[i + 1]))

        return pre_swaps + [routed_instr] + post_swaps
