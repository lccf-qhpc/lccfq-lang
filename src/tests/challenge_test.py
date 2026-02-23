"""
Filename: challenge_test.py
Author: Santiago Nunez-Corrales
Date: 2025-07-30
Version: 1.0
Description:
    Test for instruction challenging code.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest
from lccfq_lang.arch.isa import ISA
from lccfq_lang.arch.instruction import InstructionType
from lccfq_lang.arch.register import QContext, QRegister
from lccfq_lang.arch.error import MalformedInstruction, NotAllowedInContext


isa = ISA("test")


def test_valid_gate_in_circuit_context():
    instr = isa.x(tg=0)
    reg = QRegister(qubit_count=2, mapping=None, isa=None)
    challenged = reg.challenge(instr, QContext.CIRCUIT)

    assert challenged.instruction_type == InstructionType.CIRCUIT
    assert challenged.shots is None


def test_test_gate_in_circuit_context_raises():
    instr = isa.satspect(tgs=[0], shots=100)
    reg = QRegister(qubit_count=2, mapping=None, isa=None)

    with pytest.raises(NotAllowedInContext):
        reg.challenge(instr, QContext.CIRCUIT)


def test_control_instruction_in_circuit_context_raises():
    instr = isa.ftol(0.95)
    reg = QRegister(qubit_count=2, mapping=None, isa=None)

    with pytest.raises(NotAllowedInContext):
        reg.challenge(instr, QContext.CIRCUIT)


def test_valid_test_instruction_in_test_context():
    instr = isa.satspect(tgs=[0], shots=100)
    reg = QRegister(qubit_count=2, mapping=None, isa=None)

    challenged = reg.challenge(instr, QContext.TEST)
    assert challenged.instruction_type == InstructionType.TEST


def test_test_instruction_missing_shots_raises():
    instr = isa.satspect(tgs=[0], shots=None)
    reg = QRegister(qubit_count=2, mapping=None, isa=None)

    with pytest.raises(MalformedInstruction):
        reg.challenge(instr, QContext.TEST)


def test_control_instruction_in_test_context_raises():
    instr = isa.ftol(0.99)
    reg = QRegister(qubit_count=2, mapping=None, isa=None)

    with pytest.raises(NotAllowedInContext):
        reg.challenge(instr, QContext.TEST)


def test_control_instruction_with_no_context():
    instr = isa.ftol(0.99)
    reg = QRegister(qubit_count=2, mapping=None, isa=None)

    challenged = reg.challenge(instr, None)
    assert challenged.instruction_type == InstructionType.QPUSTATE