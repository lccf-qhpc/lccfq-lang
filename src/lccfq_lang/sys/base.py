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


@dataclass
class QPUConnection:
    address: str
    port: int
    username: str
    client_cert_dir: str
    server_cert: dir


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


        spec = data["qpu"]
        network_data = data["network"]

        connection = QPUConnection(
            address=network_data["address"],
            port=network_data["port"],
            username=network_data["username"],
            client_cert_dir=network_data["client_cert_dir"],
            server_cert=network_data["server_cert"]
        )

        self.name = spec["name"]
        self.location = spec["location"]
        self.topology = spec["topology"]
        self.qubit_count = int(spec["qubit_count"])
        self.qubits = spec["qubits"]
        self.couplings = spec["couplings"]
        self.exclusions = spec["exclusions"]
        self.connection = connection
