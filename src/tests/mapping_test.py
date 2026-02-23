"""
Filename: topology_test.py
Author: Santiago Nunez-Corrales
Date: 2025-08-01
Version: 1.0
Description:
    Test for the implementation of a mapping.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest

from lccfq_lang.arch.instruction import Instruction, InstructionType
from lccfq_lang.arch.isa import ISA
from lccfq_lang.arch.mapping import QPUMapping
from lccfq_lang.mach.topology import QPUTopology
from lccfq_lang.arch.error import NotEnoughQubits
from lccfq_lang.sys.base import QPUConfig


@pytest.fixture
def simple_topology_spec():
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
            "port": 1234
        }
    }


@pytest.fixture
def topology(simple_topology_spec):
    return QPUTopology(QPUConfig(simple_topology_spec))


@pytest.fixture
def isa():
    return ISA("test")


def test_qpu_mapping_initialization_success(topology):
    mapping = QPUMapping([0, 1], topology)
    assert mapping.mapping[0] in topology.qubits()
    assert mapping.mapping[1] in topology.qubits()


def test_qpu_mapping_initialization_failure(topology):
    with pytest.raises(NotEnoughQubits):
        QPUMapping([0, 1, 2, 3], topology)


def test_map_instruction(topology):
    instr = Instruction(
        symbol="cx",
        modifies_state=True,
        is_controlled=True,
        target_qubits=[1],
        control_qubits=[0],
        params=None,
        shots=None
    )
    mapping = QPUMapping([0, 1], topology)
    mapped = mapping.map(instr)

    assert mapped.symbol == "cx"
    assert mapped.target_qubits[0] in topology.qubits()
    assert mapped.control_qubits[0] in topology.qubits()
    assert mapped.is_mapped
    assert mapped.instruction_type == InstructionType.DELAYED


def test_swaps_delegation(monkeypatch, topology, isa):
    instr = Instruction(
        symbol="cx",
        modifies_state=True,
        is_controlled=True,
        target_qubits=[1],
        control_qubits=[0],
        params=None,
        shots=None
    )
    mapping = QPUMapping([0, 1], topology)
    called = {}

    def dummy_swaps(i, arch):
        called["called"] = True
        return [i]

    monkeypatch.setattr(topology, "swaps", dummy_swaps)
    result = mapping.swaps(mapping.map(instr), isa)

    assert called["called"]
    assert result[0].symbol == "cx"
