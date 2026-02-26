"""
Filename: register_test.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    Test for the implementation of registers.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import numpy as np
import pytest
from lccfq_lang.arch.isa import ISA
from lccfq_lang.arch.instruction import Instruction, InstructionType
from lccfq_lang.arch.register import QRegister, CRegister, QContext
from lccfq_lang.arch.mapping import QPUMapping
from lccfq_lang.arch.error import (
    MalformedInstruction, NotAllowedInContext, NoMeasurementsAvailable
)
from lccfq_lang.backend import QPU


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def qpu():
    return QPU(filename="src/tests/data/testing.toml")


@pytest.fixture
def isa():
    return ISA("test")


@pytest.fixture
def qreg(qpu):
    return qpu.qregister(4)


@pytest.fixture
def qreg_2q(qpu):
    return qpu.qregister(2)


# ---------------------------------------------------------------------------
# all() and but()
# ---------------------------------------------------------------------------

def test_all_returns_virtual_qubits(qreg):
    assert qreg.all() == [0, 1, 2, 3]


def test_but_none_returns_all(qreg):
    assert qreg.but() == [0, 1, 2, 3]


def test_but_excludes_specified(qreg):
    result = qreg.but(minus=[1, 3])
    assert set(result) == {0, 2}


def test_but_all_returns_empty(qreg):
    result = qreg.but(minus=[0, 1, 2, 3])
    assert result == []


# ---------------------------------------------------------------------------
# map() — virtual-to-physical qubit mapping
# ---------------------------------------------------------------------------

def test_map_single_qubit_gate(qreg, isa):
    instr = isa.x(tg=0)
    mapped = qreg.map(instr)

    assert mapped.is_mapped is True
    assert mapped.instruction_type == InstructionType.DELAYED
    assert mapped.symbol == "x"
    assert mapped.target_qubits == [0]


def test_map_two_qubit_gate(qreg, isa):
    instr = isa.cx(ct=0, tg=1)
    mapped = qreg.map(instr)

    assert mapped.is_mapped is True
    assert mapped.symbol == "cx"
    assert len(mapped.target_qubits) == 1
    assert len(mapped.control_qubits) == 1


def test_map_preserves_params(qreg, isa):
    instr = isa.rx(tg=2, params=[1.57])
    mapped = qreg.map(instr)

    assert mapped.params == [1.57]


# ---------------------------------------------------------------------------
# swaps() — swap insertion for non-adjacent qubits
# ---------------------------------------------------------------------------

def test_swaps_adjacent_no_swap(qreg, isa):
    instr = isa.cx(ct=0, tg=1)
    mapped = qreg.map(instr)
    result = qreg.swaps(mapped, isa)

    # Adjacent qubits need no swaps, just the original instruction
    assert len(result) == 1
    assert result[0].symbol == "cx"


def test_swaps_nonadjacent_inserts_swaps(qreg, isa):
    instr = isa.cx(ct=0, tg=2)
    mapped = qreg.map(instr)
    result = qreg.swaps(mapped, isa)

    # Non-adjacent on linear topology: needs pre-swap(s) + gate + post-swap(s)
    assert len(result) > 1
    symbols = [i.symbol for i in result]
    assert "swap" in symbols
    assert "cx" in symbols


def test_swaps_single_qubit_no_swap(qreg, isa):
    instr = isa.h(tg=0)
    mapped = qreg.map(instr)
    result = qreg.swaps(mapped, isa)

    assert len(result) == 1
    assert result[0].symbol == "h"


def test_swaps_measure_no_swap(qreg, isa):
    instr = isa.measure(tgs=[0])
    mapped = qreg.map(instr)
    result = qreg.swaps(mapped, isa)

    assert len(result) == 1
    assert result[0].symbol == "measure"


def test_swaps_far_qubits_more_swaps(qreg, isa):
    # Distance 1 (adjacent)
    instr_near = isa.cx(ct=0, tg=1)
    mapped_near = qreg.map(instr_near)
    result_near = qreg.swaps(mapped_near, isa)

    # Distance 3 (far on linear topology: 0-1-2-3)
    instr_far = isa.cx(ct=0, tg=3)
    mapped_far = qreg.map(instr_far)
    result_far = qreg.swaps(mapped_far, isa)

    assert len(result_far) > len(result_near)


# ---------------------------------------------------------------------------
# expand() — instruction decomposition
# ---------------------------------------------------------------------------

def test_expand_simple_gate_unchanged(qreg, isa):
    instr = isa.x(tg=0)
    expanded = qreg.expand(instr)

    assert len(expanded) == 1
    assert expanded[0].symbol == "x"


def test_expand_u2_decomposes(qreg, isa):
    instr = Instruction(
        symbol="u2",
        target_qubits=[0],
        params=[0.5, 1.0]
    )
    expanded = qreg.expand(instr)

    assert len(expanded) == 3
    assert expanded[0].symbol == "rz"
    assert expanded[1].symbol == "ry"
    assert expanded[2].symbol == "rz"
    assert expanded[0].params == [0.5]
    assert expanded[1].params == [np.pi / 2]
    assert expanded[2].params == [1.0]


def test_expand_u3_decomposes(qreg, isa):
    instr = Instruction(
        symbol="u3",
        target_qubits=[0],
        params=[0.1, 0.2, 0.3]
    )
    expanded = qreg.expand(instr)

    assert len(expanded) == 3
    assert expanded[0].symbol == "rz"
    assert expanded[1].symbol == "ry"
    assert expanded[2].symbol == "rz"
    assert expanded[0].params == [0.1]
    assert expanded[1].params == [0.2]
    assert expanded[2].params == [0.3]


def test_expand_cu_decomposes(qreg, isa):
    instr = Instruction(
        symbol="cu",
        is_controlled=True,
        target_qubits=[1],
        control_qubits=[0],
        params=[0.1, 0.2, 0.3]
    )
    expanded = qreg.expand(instr)

    assert len(expanded) == 7
    symbols = [i.symbol for i in expanded]
    assert symbols.count("rz") == 3
    assert symbols.count("ry") == 2
    assert symbols.count("cx") == 2


def test_expand_multi_measure_splits(qreg, isa):
    instr = isa.measure(tgs=[0, 1, 2])
    expanded = qreg.expand(instr)

    assert len(expanded) == 3
    assert all(e.symbol == "measure" for e in expanded)
    assert [e.target_qubits for e in expanded] == [[0], [1], [2]]


def test_expand_single_measure_unchanged(qreg, isa):
    instr = isa.measure(tgs=[0])
    expanded = qreg.expand(instr)

    assert len(expanded) == 1
    assert expanded[0].symbol == "measure"


# ---------------------------------------------------------------------------
# challenge() — context-specific validation
# ---------------------------------------------------------------------------

def test_challenge_circuit_sets_type(qreg, isa):
    instr = isa.h(tg=0)
    challenged = qreg.challenge(instr, QContext.CIRCUIT)

    assert challenged.instruction_type == InstructionType.CIRCUIT
    assert challenged.shots is None


def test_challenge_circuit_deep_copies(qreg, isa):
    instr = isa.h(tg=0)
    challenged = qreg.challenge(instr, QContext.CIRCUIT)

    # Must be a different object
    assert challenged is not instr
    # Original stays DELAYED
    assert instr.instruction_type == InstructionType.DELAYED


def test_challenge_test_sets_type(qreg, isa):
    instr = isa.x(tg=0, shots=100)
    challenged = qreg.challenge(instr, QContext.TEST)

    assert challenged.instruction_type == InstructionType.TEST


def test_challenge_test_missing_shots_raises(qreg, isa):
    instr = isa.x(tg=0)
    with pytest.raises(MalformedInstruction, match="shots"):
        qreg.challenge(instr, QContext.TEST)


def test_challenge_test_rejects_control_instruction(qreg, isa):
    instr = isa.ftol(0.95)
    with pytest.raises(NotAllowedInContext):
        qreg.challenge(instr, QContext.TEST)


def test_challenge_circuit_rejects_test_instruction(qreg, isa):
    instr = isa.satspect(tgs=[0], shots=100)
    with pytest.raises(NotAllowedInContext):
        qreg.challenge(instr, QContext.CIRCUIT)


# ---------------------------------------------------------------------------
# _is_well_formed_instruction — static validation
# ---------------------------------------------------------------------------

def test_well_formed_empty_symbol_raises():
    instr = Instruction(symbol="", target_qubits=[0])
    with pytest.raises(MalformedInstruction, match="symbol"):
        QRegister._is_well_formed_instruction(instr)


def test_well_formed_negative_qubit_raises():
    instr = Instruction(symbol="x", target_qubits=[-1])
    with pytest.raises(MalformedInstruction, match="non-negative"):
        QRegister._is_well_formed_instruction(instr)


def test_well_formed_overlapping_control_target_raises():
    instr = Instruction(
        symbol="cx", is_controlled=True,
        target_qubits=[0], control_qubits=[0]
    )
    with pytest.raises(MalformedInstruction, match="different"):
        QRegister._is_well_formed_instruction(instr)


def test_well_formed_bad_params_type_raises():
    instr = Instruction(symbol="rx", target_qubits=[0], params="bad")
    with pytest.raises(MalformedInstruction, match="parameters"):
        QRegister._is_well_formed_instruction(instr)


def test_well_formed_negative_shots_raises():
    instr = Instruction(symbol="x", target_qubits=[0], shots=-5)
    with pytest.raises(MalformedInstruction, match="shot count"):
        QRegister._is_well_formed_instruction(instr)


def test_well_formed_controlled_no_control_qubits_raises():
    instr = Instruction(
        symbol="cx", is_controlled=True,
        target_qubits=[1], control_qubits=[]
    )
    with pytest.raises(MalformedInstruction, match="control qubits"):
        QRegister._is_well_formed_instruction(instr)


# ---------------------------------------------------------------------------
# CRegister — classical register
# ---------------------------------------------------------------------------

def test_cregister_frequencies():
    creg = CRegister(size=2)
    creg.absorb({"00": 500, "01": 300, "10": 200})
    freqs = creg.frequencies()

    assert freqs["00"] == pytest.approx(0.5)
    assert freqs["01"] == pytest.approx(0.3)
    assert freqs["10"] == pytest.approx(0.2)


def test_cregister_no_data_raises():
    creg = CRegister(size=2)
    with pytest.raises(NoMeasurementsAvailable):
        creg.frequencies()


def test_cregister_zero_total():
    creg = CRegister(size=2)
    creg.absorb({"00": 0, "01": 0})
    freqs = creg.frequencies()

    assert all(v == 0.0 for v in freqs.values())
