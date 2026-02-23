"""
Filename: swap_gate_test.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    Test for instruction generation of non-parametric controled two-qubit gates.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from lccfq_lang.arch.isa import ISA
from lccfq_lang.arch.instruction import Instruction

def test_swap_gate():
    isa = ISA("lccfq")
    instr = isa.swap(tg_a=1, tg_b=2)

    assert isinstance(instr, Instruction)
    assert instr.symbol == "swap"
    assert instr.modifies_state is False
    assert instr.is_controlled is False
    assert instr.target_qubits == [2]
    assert instr.control_qubits == [1]
    assert instr.parameters is None
    assert instr.shots is None