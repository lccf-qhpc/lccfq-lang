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

@pytest.mark.parametrize("gate", ["x", "y", "z", "h", "s", "sdg", "t"])
def test_sqg_no_par_gen(gate):
    isa = ISA("lccfq")
    method = getattr(isa, gate)
    instr = method(tg=0, shots=1)

    assert isinstance(instr, Instruction)
    assert instr.symbol == gate
    assert instr.is_native is False
    assert instr.modifies_state is False
    assert instr.is_controlled is False
    assert instr.target_qubits == [0]
    assert instr.control_qubits is None
    assert instr.parameters is None
    assert instr.shots == 1