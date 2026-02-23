"""
Filename: single_par_gates_test.py
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

@pytest.mark.parametrize("gate", ["cp", "crx", "cry", "crz", "cphase", "cu"])
def test_tqcg_par_gen(gate):
    isa = ISA("lccfq")
    method = getattr(isa, gate)

    instr = None

    if gate == "cu":
        instr = method(ct = 0, tg=1, params=[0.0, 0.1, 0.2, 0.3], shots=1)
    else:
        instr = method(ct = 0, tg=1, params=[0.0], shots=1)

    assert isinstance(instr, Instruction)
    assert instr.symbol == gate
    assert instr.modifies_state is False
    assert instr.is_controlled is True
    assert instr.target_qubits == [1]
    assert instr.control_qubits == [0]
    assert instr.params == [0.0] or instr.params == [0.0, 0.1, 0.2, 0.3]
    assert instr.shots == 1