"""
Filename: transpile_tqgs_gates_test.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    Tests transpilation for two-qubit gate instructions into native gates.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest
from math import pi as PI
from lccfq_lang.arch.instruction import Instruction
from lccfq_lang.sys.factories.mach import TranspilerFactory
from lccfq_lang.mach.ir import Gate

two_qubit_gates = [
    ("cx", []),
    ("cy", []),
    ("cz", []),
    ("ch", []),
    ("cp", [PI/4]),
    ("crx", [PI/3]),
    ("cry", [PI/3]),
    ("crz", [PI/5]),
    ("cphase", [PI/2]),
    ("swap", []),
]

@pytest.mark.parametrize("symbol,params", two_qubit_gates)
def test_two_qubit_transpilation(symbol, params):
    instr = Instruction(
        symbol=symbol,
        target_qubits=[1],
        control_qubits=[0],
        parameters=params,
        shots=None,
        modifies_state=False,
        is_controlled=True
    )

    transpiler = TranspilerFactory().get(mach="pfaff_v1")()
    gates = transpiler.transpile_gate(instr)

    assert isinstance(gates, list)
    assert all(isinstance(g, Gate) for g in gates)
    assert all(g.target_qubits is not None for g in gates)
    assert symbol in transpiler._table

    for gate in gates:
        if gate.symbol in ["rx", "ry", "rz"] and gate.params is not None:
            assert all(isinstance(p, float) for p in gate.params)
