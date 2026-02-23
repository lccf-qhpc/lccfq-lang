"""
Filename: base.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file defines base, shared specifications.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from dataclasses import dataclass
from typing import List
from .error import BadQPUConfiguration


@dataclass
class QPUConnection:
    ip: str
    port: int


class QPUConfig:
    """Representation of the configuration that a QPU requires to operate inside LCCF.
    """
    name: str
    location: str
    topology: str
    qubit_count: int
    qubits: List[int]
    couplings: List[int]
    exclusions: List[int]
    connection: QPUConnection

    def __init__(self, data: dict):
        """Create a new configuration from a specification

        :param data: data containing the full specification
        """
        try:
            spec = data["qpu"]
            network_data = data["network"]
        except KeyError as e:
            raise BadQPUConfiguration("sections [qpu] and [network]", f"missing section {e}")

        required_qpu = ["name", "location", "topology", "qubit_count", "qubits", "couplings", "exclusions"]
        missing_qpu = [k for k in required_qpu if k not in spec]

        if missing_qpu:
            raise BadQPUConfiguration(f"qpu fields {required_qpu}", f"missing: {missing_qpu}")

        required_net = ["ip", "port"]
        missing_net = [k for k in required_net if k not in network_data]

        if missing_net:
            raise BadQPUConfiguration(f"network fields {required_net}", f"missing: {missing_net}")

        connection = QPUConnection(
            ip=network_data["ip"],
            port=network_data["port"]
        )

        self.name = spec["name"]
        self.location = spec["location"]
        self.topology = spec["topology"]
        self.qubit_count = int(spec["qubit_count"])
        self.qubits = spec["qubits"]
        self.couplings = spec["couplings"]
        self.exclusions = spec["exclusions"]
        self.connection = connection
