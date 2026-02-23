"""
Filename: backend_test.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    Test for the implementation of the backend object.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import toml
import pytest

from types import SimpleNamespace
from lccfq_lang.backend import QPU, QPUStatus
from lccfq_lang.sys.base import QPUConfig, QPUConnection
from lccfq_lang.mach.ir import Gate, Control
from lccfq_lang.arch.instruction import Instruction
from lccfq_lang.arch.isa import ISA
from unittest.mock import patch


@pytest.fixture
def valid_qpu_config_dict():
    return {
        "qpu": {
            "name": "pfaff_v1",
            "location": "lab42",
            "topology": "linear",
            "qubit_count": 3,
            "qubits": [0, 1, 2],
            "couplings": [(0, 1), (1, 2)],
            "exclusions": []
        },
        "network": {
            "ip": "127.0.0.1",
            "port": 5555
        }
    }


@pytest.fixture
def qpu_instance(tmp_path, valid_qpu_config_dict):
    config_path = tmp_path / "valid_qpu.toml"
    with open(config_path, "w") as f:
        toml.dump(valid_qpu_config_dict, f)

    return QPU(filename=str(config_path), last_pass="transpiled")


def test_qpu_status_enum():
    assert QPUStatus.INITIALIZED.value == 1
    assert QPUStatus.UNRESPONSIVE.value == -2
    assert QPUStatus.IDLE.name == "IDLE"


def test_qpu_initialization_valid_config(qpu_instance):
    qpu = qpu_instance
    assert qpu.config.name == "pfaff_v1"
    assert qpu.config.qubit_count == 3
    assert qpu.isa.name == "lccfq"
    assert qpu.mapping is not None


def test_qpu_has_valid_transpiler(qpu_instance):
    qpu = qpu_instance
    assert qpu.transpiler is not None


def test_qpu_map_instruction_delegation(qpu_instance):
    qpu = qpu_instance
    instr = Instruction(symbol="x", target_qubits=[0])
    mapped = qpu.map(instr)
    assert isinstance(mapped, Instruction)


def test_exec_circuit_returns_empty_dict(qpu_instance):
    qpu = qpu_instance
    dummy_circuit = [Gate(symbol="x", target_qubits=[0], control_qubits=[], params=[])]
    result = qpu.exec_circuit(dummy_circuit, shots=1)
    assert isinstance(result, dict)
    assert result == {}


def test_qpu_last_pass_stored_correctly(tmp_path, valid_qpu_config_dict):
    config_path = tmp_path / "qpu_config.toml"
    with open(config_path, "w") as f:
        toml.dump(valid_qpu_config_dict, f)

    qpu = QPU(filename=str(config_path), last_pass="mapped")
    assert qpu.last_pass == "mapped"


def test_qpu_invalid_last_pass_defaults(tmp_path, valid_qpu_config_dict):
    config_path = tmp_path / "qpu_config.toml"
    with open(config_path, "w") as f:
        toml.dump(valid_qpu_config_dict, f)

    # Should not crash on invalid pass value
    qpu = QPU(filename=str(config_path), last_pass="invalid_pass")
    assert qpu.last_pass == "invalid_pass"  # still stored, just unused


def test_bridge_method_exists(qpu_instance):
    # Not executed (it's a no-op), just confirms existence
    assert hasattr(qpu_instance, "_QPU__bridge")