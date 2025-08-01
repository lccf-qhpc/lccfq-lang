"""
Filename: base_test.py
Author: Santiago Nunez-Corrales
Date: 2025-08-01
Version: 1.0
Description:
    Test for base entities

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest
from lccfq_lang.sys.base import QPUConfig, QPUConnection


@pytest.fixture
def sample_config_dict():
    return {
        "qpu": {
            "name": "test_qpu",
            "location": "lab42",
            "topology": "linear",
            "qubit_count": 4,
            "qubits": [0, 1, 2, 3],
            "couplings": [(0, 1), (1, 2), (2, 3)],
            "exclusions": []
        },
        "network": {
            "ip": "192.168.1.10",
            "port": 4242
        }
    }


def test_qpu_config_initialization(sample_config_dict):
    config = QPUConfig(sample_config_dict)

    assert config.name == "test_qpu"
    assert config.location == "lab42"
    assert config.topology == "linear"
    assert config.qubit_count == 4
    assert config.qubits == [0, 1, 2, 3]
    assert config.couplings == [(0, 1), (1, 2), (2, 3)]
    assert config.exclusions == []

    assert isinstance(config.connection, QPUConnection)
    assert config.connection.ip == "192.168.1.10"
    assert config.connection.port == 4242