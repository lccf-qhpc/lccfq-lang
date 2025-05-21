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

from typing import List
from enum import Enum
from .error import BadTopologyType
import networkx as nx


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
        self.topo_type = topo_spec["type"]

        # TODO: translate spec into actual topology
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
        if not nx.is_empty(self.internal):
            return False

        if not nx.is_connected(self.internal):
            return False

        if self.internal.number_of_nodes() != self.internal.number_of_nodes() - 1:
            return False

        degs = [d for n, d in self.internal.degree()]

        return degs.count(1) == 2 and degs.count(2) == self.internal.number_of_nodes() - 2

    def __test(self, topo_type: QPUTopoType) -> bool:
        __dispatch = {
            topo_type.LINEAR: self.__test_linear,
        }

        return __dispatch[topo_type.LINEAR]()
