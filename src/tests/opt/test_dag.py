"""
Filename: test_dag.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Tests for circuit_to_dag and dag_to_program.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest
import networkx as nx
from lccfq_lang.arch.instruction import Instruction
from lccfq_lang.opt import circuit_to_dag, dag_to_program


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def instr(symbol, target_qubits=None, control_qubits=None, **kw):
    return Instruction(symbol=symbol, target_qubits=target_qubits,
                       control_qubits=control_qubits, **kw)


# ---------------------------------------------------------------------------
# Empty program
# ---------------------------------------------------------------------------

def test_empty_program_gives_empty_dag():
    g = circuit_to_dag([])
    assert g.number_of_nodes() == 0
    assert g.number_of_edges() == 0


def test_empty_dag_to_program():
    g = circuit_to_dag([])
    assert dag_to_program(g) == []


# ---------------------------------------------------------------------------
# Single op
# ---------------------------------------------------------------------------

def test_single_op_dag_has_one_node():
    op = instr("x", target_qubits=[0])
    g = circuit_to_dag([op])
    assert g.number_of_nodes() == 1
    assert g.number_of_edges() == 0


def test_single_op_round_trip_identity():
    op = instr("x", target_qubits=[0])
    out = dag_to_program(circuit_to_dag([op]))
    assert len(out) == 1
    assert out[0] is op


# ---------------------------------------------------------------------------
# 5-op interleaved fixture: 3 qubits
#   P[0]: h   q0
#   P[1]: cx  ctrl=q0, tg=q1
#   P[2]: rz  q1
#   P[3]: cx  ctrl=q1, tg=q2
#   P[4]: measure q2
# ---------------------------------------------------------------------------

@pytest.fixture
def program5():
    p0 = instr("h",       target_qubits=[0])
    p1 = instr("cx",      target_qubits=[1], control_qubits=[0], is_controlled=True)
    p2 = instr("rz",      target_qubits=[1], params=[0.3])
    p3 = instr("cx",      target_qubits=[2], control_qubits=[1], is_controlled=True)
    p4 = instr("measure", target_qubits=[2])
    return [p0, p1, p2, p3, p4]


def test_program5_dag_node_count(program5):
    g = circuit_to_dag(program5)
    assert g.number_of_nodes() == 5


def test_program5_dag_edges_present(program5):
    g = circuit_to_dag(program5)
    # p0 -> p1 (qubit 0), p1 -> p2 (qubit 1), p2 -> p3 (qubit 1), p3 -> p4 (qubit 2)
    assert g.has_edge(0, 1)
    assert g.has_edge(1, 2)
    assert g.has_edge(2, 3)
    assert g.has_edge(3, 4)


def test_program5_round_trip_element_wise(program5):
    out = dag_to_program(circuit_to_dag(program5))
    assert len(out) == len(program5)
    identity = [a is b for a, b in zip(program5, out)]
    assert all(identity)


# ---------------------------------------------------------------------------
# Edge qubit deduplication / multi-qubit edges
# ---------------------------------------------------------------------------

def test_edge_qubits_are_sorted_deduped():
    # Two ops on the same two qubits — edge should list both qubits once each
    p0 = instr("cx", target_qubits=[1], control_qubits=[0], is_controlled=True)
    p1 = instr("cx", target_qubits=[0], control_qubits=[1], is_controlled=True)
    g = circuit_to_dag([p0, p1])
    edge_qubits = g[0][1]["qubits"]
    assert edge_qubits == (0, 1)


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------

def test_cycle_raises_value_error():
    # Two ops on the same qubit so the DAG already has edge 0->1.
    # Adding the back-edge 1->0 creates a genuine cycle.
    p0 = instr("x", target_qubits=[0])
    p1 = instr("y", target_qubits=[0])
    g = circuit_to_dag([p0, p1])
    assert g.has_edge(0, 1), "precondition: qubit-flow edge must exist"
    # Manually introduce back-edge to create a cycle
    g.add_edge(1, 0, qubits=(0,))
    with pytest.raises(ValueError, match="cycle"):
        dag_to_program(g)


# ---------------------------------------------------------------------------
# Op with no qubits gets a node but no qubit-flow edges
# ---------------------------------------------------------------------------

def test_no_qubit_op_isolated_node():
    p0 = instr("h",       target_qubits=[0])
    p1 = Instruction(symbol="qpustate", target_qubits=None, control_qubits=None)
    p2 = instr("x",       target_qubits=[0])
    g = circuit_to_dag([p0, p1, p2])
    assert g.number_of_nodes() == 3
    # Node 1 (qpustate) has no qubit-flow edges
    assert g.in_degree(1) == 0
    assert g.out_degree(1) == 0
    # Node 0 -> Node 2 via qubit 0
    assert g.has_edge(0, 2)
