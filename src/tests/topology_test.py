"""
Filename: topology_test.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    Test for the implementation of a topology (linear).

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest
import networkx as nx

from lccfq_lang.arch.isa import ISA
from lccfq_lang.arch.instruction import Instruction
from lccfq_lang.arch.error import MalformedInstruction
from lccfq_lang.mach.topology import QPUTopology
from lccfq_lang.mach.error import QubitsNotConnected, BadTopologyType
from lccfq_lang.sys.base import QPUConfig


@pytest.fixture
def isa():
    return ISA("test")


@pytest.fixture
def linear_spec():
    return {
        "qpu": {
            "name": "pfaff_v1",
            "location": "testlab",
            "topology": "linear",
            "qubit_count": 4,
            "qubits": [0, 1, 2, 3],
            "couplings": [(0, 1), (1, 2), (2, 3)],
            "exclusions": []
        },
        "network": {
            "ip": "127.0.0.1",
            "port": 5555
        }
    }


@pytest.fixture
def disconnected_spec():
    return {
        "qpu": {
            "name": "pfaff_v1",
            "location": "testlab",
            "topology": "linear",
            "qubit_count": 3,
            "qubits": [0, 1, 2],
            "couplings": [(0, 1)],  # qubit 2 is disconnected
            "exclusions": []
        },
        "network": {
            "ip": "127.0.0.1",
            "port": 5556
        }
    }


def test_valid_linear_topology(linear_spec):
    topo = QPUTopology(QPUConfig(linear_spec))

    assert sorted(topo.qubits()) == [0, 1, 2, 3]
    assert isinstance(topo.internal, nx.Graph)
    assert nx.is_connected(topo.internal)


def test_disconnected_topology_graph_structure(disconnected_spec):
    topo = QPUTopology(QPUConfig(disconnected_spec))

    for q in disconnected_spec["qpu"]["qubits"]:
        if q not in topo.internal.nodes:
            topo.internal.add_node(q)

    assert 2 in topo.internal.nodes
    assert not nx.has_path(topo.internal, 0, 2)


def test_single_qubit_instruction_requires_no_swaps(linear_spec, isa):
    topo = QPUTopology(QPUConfig(linear_spec))
    instr = Instruction("x", target_qubits=[1])
    result = topo.swaps(instr, isa)

    assert len(result) == 1
    assert result[0].symbol == "x"


def test_two_qubit_direct_connection(linear_spec, isa):
    topo = QPUTopology(QPUConfig(linear_spec))
    instr = Instruction("cx", control_qubits=[0], target_qubits=[1])
    result = topo.swaps(instr, isa)

    assert len(result) == 1
    assert result[0].symbol == "cx"


def test_two_qubit_indirect_connection(linear_spec, isa):
    topo = QPUTopology(QPUConfig(linear_spec))
    instr = Instruction("cx", control_qubits=[0], target_qubits=[3])
    result = topo.swaps(instr, isa)
    symbols = [i.symbol for i in result]

    print(result)

    assert "cx" in symbols
    assert "swap" in symbols
    assert len(result) > 1


def test_malformed_multi_target_instruction(linear_spec, isa):
    topo = QPUTopology(QPUConfig(linear_spec))
    print(topo.real_connections)

    instr = Instruction("dummy", target_qubits=[0, 1, 2])

    with pytest.raises(MalformedInstruction):
        topo.swaps(instr, isa)


def test_disconnected_qubits_raise(disconnected_spec, isa):
    topo = QPUTopology(QPUConfig(disconnected_spec))

    for q in disconnected_spec["qpu"]["qubits"]:
        if q not in topo.internal.nodes:
            topo.internal.add_node(q)

    instr = Instruction("cx", control_qubits=[0], target_qubits=[2])

    with pytest.raises(QubitsNotConnected):
        topo.swaps(instr, isa)


def test_bad_topology_type_raises():
    bad_spec = {
        "qpu": {
            "name": "pfaff_v1",
            "location": "testlab",
            "topology": "nonsense",
            "qubit_count": 2,
            "qubits": [0, 1],
            "couplings": [(0, 1)],
            "exclusions": []
        },
        "network": {
            "ip": "127.0.0.1",
            "port": 5557
        }
    }

    with pytest.raises(BadTopologyType):
        _ = QPUTopology(QPUConfig(bad_spec))