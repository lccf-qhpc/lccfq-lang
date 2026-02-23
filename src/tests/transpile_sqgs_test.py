"""
Filename: transpile_sqgs_gates_test.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    Tests transpilation for single-qubit gate instructions into native gates.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest
from math import pi as PI
from lccfq_lang.arch.instruction import Instruction
from lccfq_lang.sys.factories.mach import TranspilerFactory
from lccfq_lang.mach.ir import Gate


single_qubit_gates = [
    ("x", []),
    ("y", []),
    ("z", []),
    ("h", []),
    ("s", []),
    ("sdg", []),
    ("t", []),
    ("tdg", []),
    ("p", [PI/2]),
    ("rx", [PI/4]),
    ("ry", [PI/3]),
    ("rz", [PI/5]),
    ("phase", [PI/6])
]

@pytest.mark.parametrize("symbol,params", single_qubit_gates)
def test_transpile_single_qubit(symbol, params):
    instr = Instruction(
        symbol=symbol,
        target_qubits=[0],
        control_qubits=None,
        params=params,
        shots=None
    )

    transpiler = TranspilerFactory().get(mach="pfaff_v1")
    gates = transpiler.transpile_gate(instr)

    print(f"{instr}")
    print("to")
    print(gates)

    assert isinstance(gates, list)
    assert all(isinstance(g, Gate) for g in gates)
    assert all(g.target_qubits == [0] for g in gates)
    assert all(isinstance(g.symbol, str) for g in gates)
