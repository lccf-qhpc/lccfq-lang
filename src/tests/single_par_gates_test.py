"""
Filename: single_nopar_gates_test.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    Test for instruction generation of non-parametric single qubit gates.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest

from lccfq_lang.arch.isa import ISA
from lccfq_lang.arch.instruction import Instruction

@pytest.mark.parametrize("gate", ["p", "rx", "ry", "rz", "u2", "u3"])
def test_sqg_par_gen(gate):
    isa = ISA("lccfq")
    method = getattr(isa, gate)

    instr = None

    if gate == "u2":
        instr = method(tg=0, params=[0.0, 0.1], shots=1)

    elif gate == "u3":
        instr = method(tg=0, params=[0.0, 0.1, 0.2], shots=1)
    else:
        instr = method(tg=0, params=[0.0], shots=1)

    assert isinstance(instr, Instruction)
    assert instr.symbol == gate
    assert instr.is_native is False
    assert instr.modifies_state is False
    assert instr.is_controlled is False
    assert instr.target_qubits == [0]
    assert instr.control_qubits is None
    assert instr.parameters == [0.0] or instr.parameters == [0.0, 0.1] or instr.parameters == [0.0, 0.1, 0.2]
    assert instr.shots == 1