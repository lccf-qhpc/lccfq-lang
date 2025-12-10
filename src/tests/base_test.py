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
from lccfq_lang.sys.error import BadQPUConfiguration


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
    assert config.connection.address == "192.168.1.10"
    assert config.connection.port == 4242


def test_missing_qpu_section_raises():
    data = {"network": {"ip": "127.0.0.1", "port": 5555}}
    with pytest.raises(BadQPUConfiguration, match="missing section"):
        QPUConfig(data)


def test_missing_network_section_raises():
    data = {"qpu": {
        "name": "q", "location": "l", "topology": "linear",
        "qubit_count": 1, "qubits": [0], "couplings": [], "exclusions": []
    }}
    with pytest.raises(BadQPUConfiguration, match="missing section"):
        QPUConfig(data)


def test_missing_qpu_fields_raises(sample_config_dict):
    del sample_config_dict["qpu"]["qubit_count"]
    del sample_config_dict["qpu"]["topology"]
    with pytest.raises(BadQPUConfiguration, match="missing.*topology.*qubit_count|missing.*qubit_count.*topology"):
        QPUConfig(sample_config_dict)


def test_missing_network_fields_raises(sample_config_dict):
    del sample_config_dict["network"]["port"]
    with pytest.raises(BadQPUConfiguration, match="missing.*port"):
        QPUConfig(sample_config_dict)


def test_empty_dict_raises():
    with pytest.raises(BadQPUConfiguration):
        QPUConfig({})