"""
Filename: two_nopar_gates_test.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    Test for instruction generation of non-parametric controled two-qubit gates.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest

from lccfq_lang.arch.isa import ISA
from lccfq_lang.arch.instruction import Instruction

@pytest.mark.parametrize("gate", ["cx", "cy", "cz", "ch"])
def test_tqcg_no_par_gen(gate):
    isa = ISA("lccfq")
    method = getattr(isa, gate)
    instr = method(ct = 0, tg=1, shots=1)

    assert isinstance(instr, Instruction)
    assert instr.symbol == gate
    assert instr.modifies_state is False
    assert instr.is_controlled is True
    assert instr.target_qubits == [1]
    assert instr.control_qubits == [0]
    assert instr.params is None
    assert instr.shots == 1